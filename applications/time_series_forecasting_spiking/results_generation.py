import logging
import os
import shutil
import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Iterable, Type, Optional

import numpy as np
import pandas as pd
import torch
import torchvision
import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from applications.time_series_forecasting_spiking.dataset import get_dataloader
import neurotorch as nt
from applications.time_series_forecasting_spiking.spikes_auto_encoder_training import (
	train_auto_encoder,
	show_single_preds,
	visualize_reconstruction,
)
from applications.util import get_optimizer
from neurotorch import Dimension, DimensionProperty, RegressionTrainer
from neurotorch.callbacks import CheckpointManager, LoadCheckpointMode
from neurotorch.callbacks.lr_schedulers import LinearLRScheduler, LRSchedulerOnMetric
from neurotorch.callbacks.training_visualization import TrainingHistoryVisualizationCallback
from neurotorch.metrics import ClassificationMetrics, RegressionMetrics
from neurotorch.modules import SequentialModel
from neurotorch.modules.layers import LayerType, LayerType2Layer, LearningType
from neurotorch.regularization import L1, L2, RegularizationList
from neurotorch.trainers import ClassificationTrainer
from neurotorch.transforms.spikes_auto_encoder import SpikesAutoEncoder
from neurotorch.utils import get_all_params_combinations, hash_params, save_params, set_seed
from neurotorch.visualisation.time_series_visualisation import VisualiseKMeans


def get_training_params_space() -> Dict[str, Any]:
	"""
	Get the parameters space for the training.
	:return: The parameters space.
	"""
	return {
		"dataset_name": [
			"timeSeries_2020_12_16_cr3_df.npy"
		],
		"n_time_steps": [
			# 8,
			16,
			# 32,
			# 64
			# 128,
			# -1
		],
		"n_encoder_steps": [
			# 8,
			16,
			32,
			64,
		],
		"n_units": [
			# 32,
			128,
			# 1024,
		],
		"encoder_type": [
			nt.LIFLayer,
			nt.ALIFLayer,
			nt.SpyLIFLayer,
		],
		"optimizer": [
			# "SGD",
			"Adam",
			# "Adamax",
			# "RMSprop",
			# "Adagrad",
			# "Adadelta",
			# "AdamW",
		],
		"learning_rate": [
			5e-5
		],
		"min_lr": [
			5e-7
		],
		"use_recurrent_connection": [
			True,
			False,
		],
		"dt": [
			1e-3,
			# 2e-2,
		],
		"smoothing_sigma": [5],
		"seed": [
			0,
		],
	}


def train_with_params(
		params: Dict[str, Any],
		n_iterations: int = 1024,
		data_folder: str = "tr_results",
		verbose: bool = False,
		show_training: bool = False,
		force_overwrite: bool = False,
		seed: int = 0,
		encoder_data_folder: Optional[str] = None,
		encoder_iterations: int = 4096,
):
	params.setdefault("smoothing_sigma", 5)
	params.setdefault("seed", seed)
	params.setdefault("optimizer", "Adam")
	params.setdefault("learning_rate", 5e-5)
	params.setdefault("min_lr", 5e-7)
	set_seed(seed)
	checkpoints_name = str(hash_params(params))
	checkpoint_folder = f"{data_folder}/{checkpoints_name}"
	os.makedirs(checkpoint_folder, exist_ok=True)
	if verbose:
		logging.info(f"Checkpoint folder: {checkpoint_folder}")
	save_params(params, os.path.join(checkpoint_folder, "params.pkl"))
	
	encoder_params = deepcopy(params)
	if encoder_data_folder is not None:
		encoder_params["data_folder"] = encoder_data_folder
		encoder_params["n_iterations"] = encoder_iterations
	auto_encoder_training_output = train_auto_encoder(**encoder_params, verbose=verbose)
	visualize_reconstruction(
		auto_encoder_training_output.dataset.data,
		auto_encoder_training_output.spikes_auto_encoder,
		filename=f"{checkpoint_folder}/figures/autoencoder_reconstruction.png",
		show=show_training,
	)
	spikes_auto_encoder = auto_encoder_training_output.spikes_auto_encoder
	auto_encoder_training_output.spikes_auto_encoder.requires_grad_(False)
	spikes_encoder = auto_encoder_training_output.spikes_auto_encoder.spikes_encoder
	spikes_encoder.spikes_layer.learning_type = nt.LearningType.NONE
	spikes_encoder.requires_grad_(False)
	spikes_decoder = auto_encoder_training_output.spikes_auto_encoder.spikes_decoder
	spikes_decoder.requires_grad_(False)
	
	dataloader = get_dataloader(units=auto_encoder_training_output.dataset.units_indexes, **params)
	spiking_foresight_steps = (params["n_time_steps"]-1)*params["n_encoder_steps"]
	network = SequentialModel(
		input_transform=[auto_encoder_training_output.spikes_auto_encoder.spikes_encoder],
		layers=[
			params["encoder_type"](
				input_size=nt.Size(
					[
						nt.Dimension(None, nt.DimensionProperty.TIME),
						nt.Dimension(params["n_units"], nt.DimensionProperty.NONE)
					]
				),
				output_size=params["n_units"],
				use_recurrent_connection=params["use_recurrent_connection"],
				learning_type=nt.LearningType.BPTT,
				name="predictor",
			),
		],
		output_transform=[auto_encoder_training_output.spikes_auto_encoder.spikes_decoder],
		name=f"network_{checkpoints_name}",
		checkpoint_folder=checkpoint_folder,
		foresight_time_steps=params.get("foresight_time_steps", spiking_foresight_steps),
	).build()
	if verbose:
		logging.info(f"\nNetwork:\n{network}")
	checkpoint_manager = nt.CheckpointManager(
		checkpoint_folder,
		metric="train_loss",
		minimise_metric=False,
		save_freq=-1,
		save_best_only=True,
	)
	callbacks = [
		# LinearLRScheduler(params.get("learning_rate", 5e-5), params.get("min_lr", 1e-7), n_iterations),
		LRSchedulerOnMetric(
			'train_loss',
			metric_schedule=np.linspace(-1.5, 0.99, 100),
			min_lr=params["min_lr"],
			retain_progress=True,
		),
		checkpoint_manager,
	]
	if show_training:
		callbacks.append(TrainingHistoryVisualizationCallback("./temp/"))
	regularization = RegularizationList([
		L2(network.parameters()),
		L1(network.parameters()),
	])
	trainer = RegressionTrainer(
		model=network,
		callbacks=callbacks,
		# regularization=regularization,
		criterion=nt.losses.PVarianceLoss(),
		optimizer=get_optimizer(params["optimizer"])(
			network.parameters(), lr=params["learning_rate"],
			maximize=True, **params.get("optimizer_params", {})
		),
		# regularization_optimizer=torch.optim.Adam(regularization.parameters(), lr=params.get("learning_rate", 2e-4)),
		lr=params["learning_rate"],
		# reg_lr=params.get("reg_lr", 2e-4),
		foresight_time_steps=params.get("foresight_time_steps", spiking_foresight_steps),
		metrics=[],
		verbose=verbose,
	)
	history = trainer.train(
		dataloader,
		n_iterations=n_iterations,
		load_checkpoint_mode=LoadCheckpointMode.LAST_ITR if not force_overwrite else None,
		force_overwrite=force_overwrite,
		exec_metrics_on_train=False,
		desc=f"Training {checkpoints_name}:{spikes_auto_encoder.encoder_type.__name__}"
		f"<{spikes_auto_encoder.n_units}u, {spikes_auto_encoder.n_encoder_steps}t>",
	)
	history.plot(save_path=f"{checkpoint_folder}/figures/training_history.png")
	try:
		network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.BEST_ITR, verbose=verbose)
	except FileNotFoundError:
		if verbose:
			logging.info("No best checkpoint found. Loading last checkpoint instead.")
		network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.LAST_ITR, verbose=verbose)
	t0, target = next(iter(dataloader))
	preds, hh = network.get_prediction_trace(
		t0, foresight_time_steps=(target.shape[1] - 1) * params["n_encoder_steps"], return_hidden_states=True
	)
	spikes_preds = hh[network.get_layer().name][-1]
	spikes = torch.squeeze(spikes_preds).detach().cpu().numpy().reshape(
		-1, spikes_auto_encoder.n_encoder_steps, spikes_auto_encoder.n_units
	)
	viz = VisualiseKMeans(
		preds.detach().cpu().numpy().squeeze(),
		shape=nt.Size([
			Dimension(preds.shape[1], dtype=DimensionProperty.TIME, name="Time Steps"),
			Dimension(preds.shape[-1], dtype=DimensionProperty.NONE, name="Neurons"),
		]),
	)
	viz.plot_timeseries_comparison(
		target, spikes,
		n_spikes_steps=params["n_encoder_steps"],
		title=f"Predictor: {spikes_auto_encoder.encoder_type.__name__}"
		f"<{spikes_auto_encoder.n_units}u, {spikes_auto_encoder.n_encoder_steps}t>",
		desc="Prediction",
		filename=f"{checkpoint_folder}/figures/forecasting_visualization.png",
		show=False,
	)
	viz.plot_timeseries(
		filename=f"{checkpoint_folder}/figures/timeseries.png",
		show=False
	)
	if verbose:
		viz.animate(
			network.get_layer().forward_weights.detach().cpu().numpy(),
			network.get_layer().dt,
			filename=f"{checkpoint_folder}/figures/animation.gif",
			show=False,
			fps=10,
		)
	viz.heatmap(
		filename=f"{checkpoint_folder}/figures/heatmap.png",
		show=False
	)
	viz.rigidplot(
		filename=f"{checkpoint_folder}/figures/rigidplot.png",
		show=False
	)
	return OrderedDict(dict(
		params=params,
		network=network,
		auto_encoder_training_output=auto_encoder_training_output,
		checkpoints_name=checkpoints_name,
		history=history,
		dataloader=dataloader,
		preds=preds.detach().cpu().numpy(),
		target=target.detach().cpu().numpy(),
		pVar=nt.losses.PVarianceLoss()(preds, target.to(preds.device)).detach().cpu().numpy(),
	))


def visualize_forecasting(
		network: nt.SequentialModel,
		spikes_auto_encoder: SpikesAutoEncoder,
		dataloader: DataLoader,
		filename: Optional[str] = None,
		show: bool = False,
) -> plt.Figure:
	t0, target = next(iter(dataloader))
	
	n_encoder_steps = spikes_auto_encoder.n_encoder_steps
	preds, hh = network.get_prediction_trace(
		t0, foresight_time_steps=(target.shape[1] - 1) * n_encoder_steps, return_hidden_states=True
	)
	spikes_preds = hh[network.get_layer().name][-1]
	
	spikes = torch.squeeze(spikes_preds).detach().cpu().numpy()
	predictions = torch.squeeze(preds.detach().cpu())
	target = torch.squeeze(target.detach().cpu())
	errors = torch.squeeze(predictions - target.to(predictions.device))**2
	mse_loss = torch.nn.MSELoss()(predictions, target.to(predictions.device))
	pVar = 1 - mse_loss / torch.var(target.to(mse_loss.device))
	fig, axes = plt.subplots(3, 1, figsize=(12, 8))
	axes[0].plot(errors.detach().cpu().numpy())
	axes[0].set_xlabel("Time [-]")
	axes[0].set_ylabel("Squared Error [-]")
	axes[0].set_title(
		f"Predictor: {spikes_auto_encoder.encoder_type.__name__}"
		f"<{spikes_auto_encoder.n_units}u, {spikes_auto_encoder.n_encoder_steps}t>, "
		f"pVar: {pVar.detach().cpu().item():.3f}"
	)
	
	mean_errors = torch.mean(errors, dim=0)
	mean_error_sort, indices = torch.sort(mean_errors)
	predictions = torch.squeeze(predictions).numpy().T
	target = torch.squeeze(target).numpy().T
	
	best_idx, worst_idx = indices[0], indices[-1]
	spikes = spikes.reshape(-1, spikes_auto_encoder.n_encoder_steps, spikes_auto_encoder.n_units)
	show_single_preds(
		spikes_auto_encoder, axes[1], predictions[best_idx], target[best_idx], spikes[:, :, best_idx],
		title="Best Prediction"
	)
	show_single_preds(
		spikes_auto_encoder, axes[2], predictions[worst_idx], target[worst_idx], spikes[:, :, worst_idx],
		title="Worst Prediction"
	)
	
	fig.set_tight_layout(True)
	if filename is not None:
		os.makedirs(os.path.dirname(filename), exist_ok=True)
		fig.savefig(filename)
	if show:
		plt.show()
	plt.close(fig)
	return fig


def train_all_params(
		training_params: Dict[str, Any] = None,
		n_iterations: int = 100,
		data_folder: str = "tr_results",
		verbose: bool = False,
		rm_data_folder_and_restart_all_training: bool = False,
		force_overwrite: bool = False,
		skip_if_exists: bool = False,
		encoder_data_folder: Optional[str] = None,
		encoder_iterations: int = 4096,
):
	"""
	Train the network with all the parameters.
	
	:param n_iterations: The number of iterations to train the network.
	:param verbose: If True, print the progress.
	:param data_folder: The folder where to save the data.
	:param training_params: The parameters to use for the training.
	:param rm_data_folder_and_restart_all_training: If True, remove the data folder and restart all the training.
	:param force_overwrite: If True, overwrite and restart non-completed training.
	:param skip_if_exists: If True, skip the training if the results already in the results dataframe.
	:param encoder_data_folder: The folder where to load and save the encoder data.
	:param encoder_iterations: The number of iterations to train the encoder.
	:return: The results of the training.
	"""
	warnings.filterwarnings("ignore", category=Warning)
	if rm_data_folder_and_restart_all_training and os.path.exists(data_folder):
		shutil.rmtree(data_folder)
	os.makedirs(data_folder, exist_ok=True)
	results_path = os.path.join(data_folder, "results.csv")
	if training_params is None:
		training_params = get_training_params_space()

	all_params_combinaison_dict = get_all_params_combinations(training_params)
	columns = [
		'checkpoints',
		*list(training_params.keys()),
		'pVar',
	]

	# load dataframe if exists
	try:
		df = pd.read_csv(results_path)
	except FileNotFoundError:
		df = pd.DataFrame(columns=columns)

	with tqdm.tqdm(all_params_combinaison_dict, desc="Training all the parameters", position=0) as p_bar:
		for i, params in enumerate(p_bar):
			if str(hash_params(params)) in df["checkpoints"].values and skip_if_exists:
				continue
			# p_bar.set_description(f"Training {params}")
			try:
				result = train_with_params(
					params,
					n_iterations=n_iterations,
					data_folder=data_folder,
					verbose=verbose,
					show_training=False,
					force_overwrite=force_overwrite,
					encoder_data_folder=encoder_data_folder,
					encoder_iterations=encoder_iterations,
				)
				params.update(result["params"])
				if result["checkpoints_name"] in df["checkpoints"].values:
					# remove from df if already exists
					df = df[df["checkpoints"] != result["checkpoints_name"]]
				df = pd.concat([df, pd.DataFrame(
					dict(
						checkpoints=[result["checkpoints_name"]],
						**{k: [v] for k, v in params.items()},
						pVar=[result["pVar"]],
					))],  ignore_index=True,
				)
				df.to_csv(results_path, index=False)
				p_bar.set_postfix(
					params=params,
					pVar=result["pVar"],
				)
			except Exception as e:
				logging.error(e)
				continue
	return df



