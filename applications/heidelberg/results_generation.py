import logging
import os
import shutil
import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Iterable, Type

import pandas as pd
import torch
import tqdm

from applications.heidelberg.dataset import get_dataloaders
import neurotorch as nt
from neurotorch import Dimension, DimensionProperty
from neurotorch.callbacks import CheckpointManager, LoadCheckpointMode
from neurotorch.callbacks.training_visualization import TrainingHistoryVisualizationCallback
from neurotorch.metrics import ClassificationMetrics
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
		"n_hidden_neurons": [
			# 16,
			# 32,
			# 64,
			# [64, 64],
			# 128,
			256,
			# [32, 32],
			# 32
		],
		"hidden_layer_type": [
			LayerType.LIF,
			LayerType.ALIF,
			LayerType.SpyLIF,
		],
		"readout_layer_type": [
			LayerType.LI,
			LayerType.SpyLI,
		],
		"use_recurrent_connection": [
			False,
			True
		],
		# "learn_beta": [
		# 	True,
		# 	False
		# ],
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
		n_iterations: int = 100,
		batch_size: int = 256,
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

	dataloaders = get_dataloaders(
		batch_size=batch_size,
		train_val_split_ratio=params.get("train_val_split_ratio", 0.95),
	)
	n_features = dataloaders["test"].dataset.n_units
	n_hidden_neurons = params["n_hidden_neurons"]
	if not isinstance(n_hidden_neurons, Iterable):
		n_hidden_neurons = [n_hidden_neurons]

	hidden_layers = [
		LayerType2Layer[params["hidden_layer_type"]](
			input_size=n_hidden_neurons[i],
			output_size=n,
			**params
		)
		for i, n in enumerate(n_hidden_neurons[1:])
	] if len(n_hidden_neurons) > 1 else []
	network = SequentialModel(
		layers=[
			LayerType2Layer[params["hidden_layer_type"]](
				input_size=nt.Size([
					Dimension(None, DimensionProperty.TIME),
					Dimension(n_features, DimensionProperty.NONE)
				]),
				output_size=n_hidden_neurons[0],
				**params
			),
			*hidden_layers,
			LayerType2Layer[params["readout_layer_type"]](output_size=dataloaders["test"].dataset.n_classes),
		],
		name=f"heidelberg_network_{checkpoints_name}",
		checkpoint_folder=checkpoint_folder,
		foresight_time_steps=params.get("foresight_time_steps", 0),
	)
	network.build()
	if verbose:
		logging.info(f"\nNetwork:\n{network}")
	checkpoint_manager = CheckpointManager(checkpoint_folder, metric="val_accuracy", minimise_metric=False)
	save_params(params, os.path.join(checkpoint_folder, "params.pkl"))
	callbacks = [checkpoint_manager, ]
	if show_training:
		callbacks.append(TrainingHistoryVisualizationCallback("./temp/"))
	regularization = RegularizationList([
		L2(network.parameters()),
		L1(network.parameters()),
	])
	trainer = ClassificationTrainer(
		model=network,
		callbacks=callbacks,
		regularization=regularization,
		optimizer=get_optimizer(params.get("optimizer", "adam"))(
			network.parameters(), lr=params.get("learning_rate", 2e-4), **params.get("optimizer_params", {})
		),
		# regularization_optimizer=torch.optim.Adam(regularization.parameters(), lr=params.get("learning_rate", 2e-4)),
		lr=params.get("learning_rate", 2e-4),
		reg_lr=params.get("reg_lr", 2e-4),
		verbose=verbose,
	)
	history = trainer.train(
		dataloaders["train"],
		dataloaders["val"],
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

	predictions = {
		k: ClassificationMetrics.compute_y_true_y_pred(network, dataloader, verbose=verbose, desc=f"{k} predictions")
		for k, dataloader in dataloaders.items()
	}
	return OrderedDict(dict(
		network=network,
		checkpoints_name=checkpoints_name,
		history=history,
		accuracies={
			k: ClassificationMetrics.accuracy(network, y_true=y_true, y_pred=y_pred)
			for k, (y_true, y_pred) in predictions.items()
		},
		precisions={
			k: ClassificationMetrics.precision(network, y_true=y_true, y_pred=y_pred)
			for k, (y_true, y_pred) in predictions.items()
		},
		recalls={
			k: ClassificationMetrics.recall(network, y_true=y_true, y_pred=y_pred)
			for k, (y_true, y_pred) in predictions.items()
		},
		f1s={
			k: ClassificationMetrics.f1(network, y_true=y_true, y_pred=y_pred)
			for k, (y_true, y_pred) in predictions.items()
		},
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



