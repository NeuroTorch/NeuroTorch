import logging
import os
import shutil
import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Iterable, Type

import numpy as np
import pandas as pd
import torch
import torchvision
import tqdm
from torchvision.transforms import Compose

from applications.fit_wilson_cowan_with_lif.dataset import get_dataloader
import neurotorch as nt
from neurotorch import Dimension, DimensionProperty, RegressionTrainer
from neurotorch.callbacks import CheckpointManager, LoadCheckpointMode
from neurotorch.callbacks.training_visualization import TrainingHistoryVisualizationCallback
from neurotorch.metrics import ClassificationMetrics, RegressionMetrics
from neurotorch.modules import SequentialModel
from neurotorch.modules.layers import LayerType, LayerType2Layer, LearningType
from neurotorch.regularization import L1, L2, RegularizationList
from neurotorch.trainers import ClassificationTrainer
from neurotorch.utils import get_all_params_combinations, hash_params, save_params


def get_training_params_space() -> Dict[str, Any]:
	"""
	Get the parameters space for the training.
	:return: The parameters space.
	"""
	return {
		"n_steps": [
			# 8,
			# 16,
			# 32,
			# 64
			100,
			# 1_000
		],
		"hidden_layer_type": [
			LayerType.LIF,
			LayerType.ALIF,
			LayerType.SpyLIF,
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
			# 1e-2,
			# 1e-3,
			2e-4,
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
		t_0: np.ndarray,
		forward_weights: np.ndarray,
		mu: np.ndarray,
		r: float,
		tau: float,
		n_iterations: int = 100,
		data_folder: str = "tr_results",
		verbose: bool = False,
		show_training: bool = False,
		force_overwrite: bool = False,
		seed: int = 42,
):
	torch.manual_seed(seed)
	checkpoints_name = str(hash_params(params))
	checkpoint_folder = f"{data_folder}/{checkpoints_name}"
	os.makedirs(checkpoint_folder, exist_ok=True)
	if verbose:
		logging.info(f"Checkpoint folder: {checkpoint_folder}")
	
	assert len(t_0) == forward_weights.shape[0] == forward_weights.shape[1] == params["n_units"]
	
	dataloader = get_dataloader(
		n_steps=params["n_steps"],
		dt=params["dt"],
		t_0=t_0,
		forward_weights=forward_weights,
		spikes_transform=params["encoder"](
			n_steps=params["n_encoder_steps"],
			n_units=params["n_units"],
			dt=params["dt"],
		),
		mu=mu,
		r=r,
		tau=tau,
	)
	spiking_foresight_steps = (params["n_steps"]-1)*params["n_encoder_steps"]
	spiking_output_trace_steps = params["n_steps"]*params["n_encoder_steps"]
	network = SequentialModel(
		layers=[
			LayerType2Layer[params["hidden_layer_type"]](
				input_size=nt.Size([
					Dimension(None, DimensionProperty.TIME),
					Dimension(dataloader.dataset.n_units, DimensionProperty.NONE)
				]),
				output_size=dataloader.dataset.n_units,
				use_recurrent_connection=False,
				**params
			),
		],
		output_transform=[torch.nn.Sequential(
			torchvision.ops.Permute([0, 2, 1]),
			torch.nn.Conv1d(
				params["n_units"], params["n_units"], params["n_encoder_steps"],
				stride=params["n_encoder_steps"]
			),
			torchvision.ops.Permute([0, 2, 1]),
		)],
		name=f"network_{checkpoints_name}",
		checkpoint_folder=checkpoint_folder,
		foresight_time_steps=params.get("foresight_time_steps", spiking_foresight_steps),
	)
	network.build()
	if verbose:
		logging.info(f"\nNetwork:\n{network}")
	checkpoint_manager = CheckpointManager(checkpoint_folder, metric="train_loss", minimise_metric=True)
	save_params(params, os.path.join(checkpoint_folder, "params.pkl"))
	callbacks = [checkpoint_manager, ]
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
		optimizer=get_optimizer(params.get("optimizer", "adam"))(
			network.parameters(), lr=params.get("learning_rate", 2e-4), **params.get("optimizer_params", {})
		),
		# regularization_optimizer=torch.optim.Adam(regularization.parameters(), lr=params.get("learning_rate", 2e-4)),
		lr=params.get("learning_rate", 2e-4),
		# reg_lr=params.get("reg_lr", 2e-4),
		foresight_time_steps=params["n_steps"],
		verbose=verbose,
	)
	history = trainer.train(
		dataloader,
		n_iterations=n_iterations,
		load_checkpoint_mode=LoadCheckpointMode.LAST_ITR if not force_overwrite else None,
		force_overwrite=force_overwrite,
		exec_metrics_on_train=False,
	)
	try:
		network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.BEST_ITR, verbose=verbose)
	except FileNotFoundError:
		if verbose:
			logging.info("No best checkpoint found. Loading last checkpoint instead.")
		network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.LAST_ITR, verbose=verbose)

	y_true, y_pred = RegressionMetrics.compute_y_true_y_pred(network, dataloader, verbose=verbose, desc=f"predictions")
	return OrderedDict(dict(
		network=network,
		checkpoints_name=checkpoints_name,
		history=history,
		pVar=RegressionMetrics.p_var(network, y_true=y_true, y_pred=y_pred),
	))


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


