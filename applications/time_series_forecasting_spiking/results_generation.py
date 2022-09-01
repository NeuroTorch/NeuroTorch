import logging
import os
import shutil
import time
import warnings
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any, Dict, Iterable, Type, Optional, Union, Tuple

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
	visualize_reconstruction, AutoEncoderTrainingOutput,
)
from applications.util import get_optimizer, get_regularization
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
			4,
			8,
			16,
			32,
			64,
			128,
			256,
			# -1
		],
		"n_encoder_steps": [
			# 8,
			16,
			# 32,
			# 64,
		],
		"n_units": [
			# 32,
			128,
			# 1024,
		],
		"encoder_type": [
			# nt.LIFLayer,
			# nt.ALIFLayer,
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
			# True,
			False,
		],
		"dt": [
			1e-3,
			# 2e-2,
		],
		"smoothing_sigma": [
			# 0,
			5,
			# 10,
		],
		"seed": [
			0,
		],
	}


def set_default_params(params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
	params.setdefault("smoothing_sigma", 5)
	params.setdefault("seed", kwargs["seed"])
	params.setdefault("optimizer", "Adam")
	params.setdefault("learning_rate", 5e-5)
	params.setdefault("min_lr", 5e-7)
	params.setdefault("reg_lr", 1e-7)
	params.setdefault("reg", None)
	params.setdefault("dataset_length", 1)
	return params


def train_and_get_autoencoder(
		params: Dict[str, Any],
		*,
		show_training: bool,
		checkpoint_folder: str,
		encoder_data_folder: Optional[str],
		encoder_iterations: int,
		verbose: bool = True,
		**kwargs
) -> AutoEncoderTrainingOutput:
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
	auto_encoder_training_output.spikes_auto_encoder.requires_grad_(False)
	spikes_encoder = auto_encoder_training_output.spikes_auto_encoder.spikes_encoder
	spikes_encoder.spikes_layer.learning_type = nt.LearningType.NONE
	spikes_encoder.requires_grad_(False)
	spikes_decoder = auto_encoder_training_output.spikes_auto_encoder.spikes_decoder
	spikes_decoder.requires_grad_(False)
	return auto_encoder_training_output


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
		batch_size: int = 32,
		save_best_only: bool = True,
):
	params = set_default_params(params, seed=seed)
	set_seed(seed)
	checkpoints_name = str(hash_params(params))
	checkpoint_folder = f"{data_folder}/{checkpoints_name}"
	os.makedirs(checkpoint_folder, exist_ok=True)
	if verbose:
		logging.info(f"Checkpoint folder: {checkpoint_folder}")
	save_params(params, os.path.join(checkpoint_folder, "params.pkl"))
	
	auto_encoder_training_output = train_and_get_autoencoder(
		params,
		show_training=show_training,
		checkpoint_folder=checkpoint_folder,
		encoder_data_folder=encoder_data_folder,
		encoder_iterations=encoder_iterations,
	)
	spikes_auto_encoder = auto_encoder_training_output.spikes_auto_encoder
	
	dataloader = get_dataloader(
		units=auto_encoder_training_output.dataset.units_indexes, batch_size=batch_size, **params
	)
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
	initial_weights = deepcopy(network.get_layer().forward_weights.detach().cpu())
	if verbose:
		logging.info(f"\nNetwork:\n{network}")
	checkpoint_manager = nt.CheckpointManager(
		checkpoint_folder,
		metric="train_loss",
		minimise_metric=False,
		save_freq=-1,
		save_best_only=save_best_only,
		start_save_at=int(0.98*n_iterations),
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
	regularization = get_regularization(params["reg"], [network.get_layer().forward_weights])
	if regularization is None:
		regularization_optimizer = None
	else:
		regularization_optimizer = torch.optim.Adam(regularization.parameters(), lr=params["reg_lr"])
	trainer = RegressionTrainer(
		model=network,
		callbacks=callbacks,
		regularization=regularization,
		criterion=nt.losses.PVarianceLoss(),
		optimizer=get_optimizer(params["optimizer"])(
			network.parameters(), lr=params["learning_rate"],
			maximize=True, **params.get("optimizer_params", {})
		),
		regularization_optimizer=regularization_optimizer,
		lr=params["learning_rate"],
		reg_lr=params["reg_lr"],
		foresight_time_steps=params.get("foresight_time_steps", spiking_foresight_steps),
		metrics=[],
		verbose=verbose,
	)
	start_time = time.time()
	history = trainer.train(
		dataloader,
		n_iterations=n_iterations,
		load_checkpoint_mode=LoadCheckpointMode.LAST_ITR if not force_overwrite else None,
		force_overwrite=force_overwrite,
		exec_metrics_on_train=False,
		desc=f"Training {checkpoints_name}:{spikes_auto_encoder.encoder_type.__name__}"
		f"<{spikes_auto_encoder.n_units}u, {spikes_auto_encoder.n_encoder_steps}t>",
	)
	training_time = time.time() - start_time
	history.plot(save_path=f"{checkpoint_folder}/figures/training_history.png")
	try:
		network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.BEST_ITR, verbose=verbose)
	except FileNotFoundError:
		if verbose:
			logging.info("No best checkpoint found. Loading last checkpoint instead.")
		network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.LAST_ITR, verbose=verbose)
	preds_targets_spikes = compute_preds_targets_spikes(
		network,
		get_dataloader(
			units=auto_encoder_training_output.dataset.units_indexes, batch_size=batch_size, shuffle=False, **params
		),
		verbose=verbose,
		desc="Compute predictions, spikes and targets",
	)
	make_figures(
		params, network,
		preds_targets_spikes=preds_targets_spikes,
		auto_encoder_training_output=auto_encoder_training_output,
		initial_weights=initial_weights,
		checkpoint_folder=checkpoint_folder,
		verbose=verbose,
	)
	
	return OrderedDict(dict(
		params=params,
		network=network,
		auto_encoder_training_output=auto_encoder_training_output,
		checkpoint_folder=checkpoint_folder,
		checkpoints_name=checkpoints_name,
		history=history,
		training_time=training_time,
		dataloader=dataloader,
		predictions=preds_targets_spikes["predictions"],
		targets=preds_targets_spikes["targets"],
		pVar=nt.to_numpy(nt.losses.PVarianceLoss()(
			nt.to_tensor(preds_targets_spikes["predictions"]).to("cpu"),
			nt.to_tensor(preds_targets_spikes["targets"]).to("cpu")
		)),
	))


def compute_preds_targets_spikes(
		network: SequentialModel,
		dataloader: DataLoader,
		device: Optional[torch.device] = None,
		verbose: bool = False,
		desc: Optional[str] = None,
		p_bar_position: int = 0,
) -> Dict[str, np.ndarray]:
	if device is not None:
		network.to(device)
	network.eval()
	predictions = []
	targets = []
	spikes = []
	with torch.no_grad():
		for i, (x, y_true) in tqdm.tqdm(
				enumerate(dataloader), total=len(dataloader),
				desc=desc, disable=not verbose, position=p_bar_position,
		):
			x = x.to(network.device)
			y_true = y_true.to(network.device)
			preds, hh = network.get_prediction_trace(x, return_hidden_states=True)
			spikes_preds = hh[network.get_layer().name][-1]
			n_encoder_steps = int(spikes_preds.shape[1] / preds.shape[1])
			spikes_preds = nt.to_numpy(spikes_preds).reshape(
				spikes_preds.shape[0], -1, n_encoder_steps, y_true.shape[-1]
			)
			predictions.extend(nt.to_numpy(preds))
			targets.extend(nt.to_numpy(y_true))
			spikes.extend(nt.to_numpy(spikes_preds))
			
	predictions = np.asarray(predictions)
	targets = np.asarray(targets)
	spikes = np.asarray(spikes)
	return dict(predictions=predictions, targets=targets, spikes=spikes)


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


def visualize_init_final_weights(
		initial_wights: Any,
		final_weights: Any,
		filename: Optional[str] = None,
		show: bool = False,
) -> plt.Figure:
	from mpl_toolkits.axes_grid1.inset_locator import inset_axes
	initial_wights, final_weights = nt.to_numpy(initial_wights), nt.to_numpy(final_weights)
	initial_dale = nt.to_numpy(nt.DaleLaw([nt.to_tensor(initial_wights)], seed=0)())
	final_dale = nt.to_numpy(nt.DaleLaw([nt.to_tensor(final_weights)], seed=0)())
	fig, axes = plt.subplots(1, 2, figsize=(12, 8))
	im = axes[0].imshow(initial_wights, cmap="RdBu_r")
	axes[0].set_title(f"Initial Weights (Dale loss = {initial_dale:.3f})")
	axes[1].imshow(final_weights, cmap=im.cmap, extent=im.get_extent())
	axes[1].set_title(f"Final Weights (Dale loss = {final_dale:.3f})")
	cax = inset_axes(
		axes[1],
		width="5%",
		height="100%",
		loc='lower left',
		bbox_to_anchor=(1.05, 0.0, 1.0, 1.0),
		bbox_transform=axes[1].transAxes,
		borderpad=0,
	)
	fig.colorbar(im, cax=cax)
	fig.set_tight_layout(False)
	if filename is not None:
		os.makedirs(os.path.dirname(filename), exist_ok=True)
		fig.savefig(filename)
	if show:
		plt.show()
	plt.close(fig)
	return fig


def make_figures(
		params: Dict[str, Any],
		network: nt.SequentialModel,
		*,
		auto_encoder_training_output,
		preds_targets_spikes: dict,
		initial_weights,
		checkpoint_folder: str,
		verbose: bool,
):
	spikes_auto_encoder = auto_encoder_training_output.spikes_auto_encoder
	preds = preds_targets_spikes["predictions"].reshape(
		-1, spikes_auto_encoder.n_units
	)
	spikes = preds_targets_spikes["spikes"].reshape(
		-1, spikes_auto_encoder.n_encoder_steps, spikes_auto_encoder.n_units
	)
	targets = preds_targets_spikes["targets"].reshape(
		-1, spikes_auto_encoder.n_units
	)
	viz = VisualiseKMeans(
		preds,
		shape=nt.Size(
			[
				Dimension(preds.shape[0], dtype=DimensionProperty.TIME, name="Time Steps"),
				Dimension(preds.shape[-1], dtype=DimensionProperty.NONE, name="Neurons"),
			]
		),
	)
	viz.plot_timeseries_comparison(
		targets, spikes,
		n_spikes_steps=params["n_encoder_steps"],
		title=f"Predictor: {spikes_auto_encoder.encoder_type.__name__}"
		f"<{spikes_auto_encoder.n_units}u, {spikes_auto_encoder.n_encoder_steps}t>",
		desc="Prediction",
		filename=f"{checkpoint_folder}/figures/forecasting_visualization.png",
		show=False,
	)
	viz.plot_timeseries(
		filename=f"{checkpoint_folder}/figures/preds_timeseries.png",
		show=False
	)
	if verbose:
		viz.animate(
			network.get_layer().forward_weights.detach().cpu().numpy(),
			network.get_layer().dt,
			filename=f"{checkpoint_folder}/figures/preds_animation.gif",
			show=False,
			fps=10,
		)
	viz.heatmap(
		filename=f"{checkpoint_folder}/figures/preds_heatmap.png",
		show=False
	)
	viz.rigidplot(
		filename=f"{checkpoint_folder}/figures/preds_rigidplot.png",
		show=False
	)
	visualize_init_final_weights(
		initial_weights,
		network.get_layer().forward_weights,
		filename=f"{checkpoint_folder}/figures/init_final_weights.png",
		show=False
	)


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
				training_time = result["training_time"]
				if result["checkpoints_name"] in df["checkpoints"].values:
					training_time = df.loc[df["checkpoints"] == result["checkpoints_name"], "training_time"].values[0]
					# remove from df if already exists
					df = df[df["checkpoints"] != result["checkpoints_name"]]
				df = pd.concat([df, pd.DataFrame(
					dict(
						checkpoints=[result["checkpoints_name"]],
						**{k: [v] for k, v in params.items()},
						training_time=training_time,
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



