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
from neurotorch import Dimension, DimensionProperty, RegressionTrainer
from neurotorch.callbacks import CheckpointManager, LoadCheckpointMode
from neurotorch.callbacks.lr_schedulers import LinearLRScheduler
from neurotorch.callbacks.training_visualization import TrainingHistoryVisualizationCallback
from neurotorch.metrics import ClassificationMetrics, RegressionMetrics
from neurotorch.modules import SequentialModel
from neurotorch.modules.layers import LayerType, LayerType2Layer, LearningType
from neurotorch.regularization import L1, L2, RegularizationList
from neurotorch.trainers import ClassificationTrainer
from neurotorch.transforms.spikes_auto_encoder import SpikesAutoEncoder
from neurotorch.utils import get_all_params_combinations, hash_params, save_params, set_seed


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
			8,
			16,
			32,
			64,
		],
		"n_units": [
			32,
			128,
			1024,
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
			"AdamW",
		],
		"learning_rate": [
			5e-5
		],
		"use_recurrent_connection": [
			True,
			False,
		],
		"dt": [
			1e-3,
			2e-2
		],
		"seed": [
			0,
		],
	}


def get_optimizer(optimizer_name: str) -> Type[torch.optim.Optimizer]:
	name_to_opt = {
		"sgd": torch.optim.SGD,
		"adam": torch.optim.Adam,
		"adamax": torch.optim.Adamax,
		"rmsprop": torch.optim.RMSprop,
		"adagrad": torch.optim.Adagrad,
		"adadelta": torch.optim.Adadelta,
		"adamw": torch.optim.AdamW,
	}
	return name_to_opt[optimizer_name.lower()]


def train_with_params(
		params: Dict[str, Any],
		n_iterations: int = 1024,
		data_folder: str = "tr_results",
		verbose: bool = False,
		show_training: bool = False,
		force_overwrite: bool = False,
		seed: int = 0,
):
	set_seed(seed)
	checkpoints_name = str(hash_params(params))
	checkpoint_folder = f"{data_folder}/{checkpoints_name}"
	os.makedirs(checkpoint_folder, exist_ok=True)
	if verbose:
		logging.info(f"Checkpoint folder: {checkpoint_folder}")
	save_params(params, os.path.join(checkpoint_folder, "params.pkl"))
	
	auto_encoder_training_output = train_auto_encoder(**params)
	visualize_reconstruction(
		auto_encoder_training_output.dataset.data,
		auto_encoder_training_output.spikes_auto_encoder,
		filename=f"{checkpoint_folder}/autoencoder_reconstruction.png",
		show=show_training,
	)
	spikes_auto_encoder = auto_encoder_training_output.spikes_auto_encoder
	auto_encoder_training_output.spikes_auto_encoder.requires_grad_(False)
	
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
	)
	network.build()
	if verbose:
		logging.info(f"\nNetwork:\n{network}")
	checkpoint_manager = nt.CheckpointManager(
		checkpoint_folder,
		metric="train_loss",
		minimise_metric=False,
		save_freq=-1,
		save_best_only=False,
	)
	callbacks = [checkpoint_manager, LinearLRScheduler(5e-5, 1e-7, n_iterations)]
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
		optimizer=get_optimizer(params.get("optimizer", "adam"))(
			network.parameters(), lr=params.get("learning_rate", 5e-5),
			maximize=True, **params.get("optimizer_params", {})
		),
		# regularization_optimizer=torch.optim.Adam(regularization.parameters(), lr=params.get("learning_rate", 2e-4)),
		lr=params.get("learning_rate", 5e-5),
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
		desc=f"Training {spikes_auto_encoder.encoder_type.__name__}"
		f"<{spikes_auto_encoder.n_units}u, {spikes_auto_encoder.n_encoder_steps}t>",
	)
	try:
		network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.BEST_ITR, verbose=verbose)
	except FileNotFoundError:
		if verbose:
			logging.info("No best checkpoint found. Loading last checkpoint instead.")
		network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.LAST_ITR, verbose=verbose)
	visualize_forecasting(
		network, spikes_auto_encoder, dataloader,
		filename=f"{checkpoint_folder}/forecasting_visualization.png",
		show=show_training,
	)
	y_true, y_pred = RegressionMetrics.compute_y_true_y_pred(network, dataloader, verbose=verbose, desc=f"predictions")
	return OrderedDict(dict(
		network=network,
		auto_encoder_training_output=auto_encoder_training_output,
		checkpoints_name=checkpoints_name,
		history=history,
		pVar=RegressionMetrics.p_var(network, y_true=y_true, y_pred=y_pred),
	))


def visualize_forecasting(
		network: nt.SequentialModel,
		spikes_auto_encoder: SpikesAutoEncoder,
		dataloader: DataLoader,
		filename: Optional[str] = None,
		show: bool = False,
):
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
		f"pVar: {pVar.detach().cpu().item():.3f}, "
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
		fig.savefig(filename)
	if show:
		plt.show()
	plt.close(fig)


def train_all_params(
		training_params: Dict[str, Any] = None,
		n_iterations: int = 100,
		batch_size: int = 256,
		data_folder: str = "tr_results",
		verbose: bool = False,
		rm_data_folder_and_restart_all_training: bool = False,
		force_overwrite: bool = False,
		skip_if_exists: bool = False,
):
	"""
	Train the network with all the parameters.
	
	:param n_iterations: The number of iterations to train the network.
	:param batch_size: The batch size to use.
	:param verbose: If True, print the progress.
	:param data_folder: The folder where to save the data.
	:param training_params: The parameters to use for the training.
	:param rm_data_folder_and_restart_all_training: If True, remove the data folder and restart all the training.
	:param force_overwrite: If True, overwrite and restart non-completed training.
	:param skip_if_exists: If True, skip the training if the results already in the results dataframe.
	:return: The results of the training.
	"""
	warnings.filterwarnings("ignore", category=UserWarning)
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
		'train_accuracy', 'val_accuracy', 'test_accuracy',
		'train_precision', 'val_precision', 'test_precision',
		'train_recall', 'val_recall', 'test_recall',
		'train_f1', 'val_f1', 'test_f1',
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
					batch_size=batch_size,
					data_folder=data_folder,
					verbose=verbose,
					show_training=False,
					force_overwrite=force_overwrite,
				)
				if str(hash_params(params)) in df["checkpoints"].values:
					# remove from df if already exists
					df = df[df["checkpoints"] != result["checkpoints_name"]]
				df = pd.concat([df, pd.DataFrame(
					dict(
						checkpoints=[result["checkpoints_name"]],
						**{k: [v] for k, v in params.items()},
						# accuracies
						train_accuracy=[result["accuracies"]["train"]],
						val_accuracy=[result["accuracies"]["val"]],
						test_accuracy=[result["accuracies"]["test"]],
						# precisions
						train_precision=[result["precisions"]["train"]],
						val_precision=[result["precisions"]["val"]],
						test_precision=[result["precisions"]["test"]],
						# recalls
						train_recall=[result["recalls"]["train"]],
						val_recall=[result["recalls"]["val"]],
						test_recall=[result["recalls"]["test"]],
						# f1s
						train_f1=[result["f1s"]["train"]],
						val_f1=[result["f1s"]["val"]],
						test_f1=[result["f1s"]["test"]],
					))],  ignore_index=True,
				)
				df.to_csv(results_path, index=False)
				p_bar.set_postfix(
					params=params,
					train_accuracy=result["accuracies"]['train'],
					val_accuracy=result["accuracies"]['val'],
					test_accuracy=result["accuracies"]['test']
				)
			except Exception as e:
				logging.error(e)
				continue
	return df



