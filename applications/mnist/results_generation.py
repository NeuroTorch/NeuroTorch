import itertools
import logging
import os
import pickle
import shutil
import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Iterable, List

import norse
import numpy as np
import pandas as pd
import psutil
import torch
import tqdm
from torchvision.transforms import Compose, Lambda

from applications.mnist.dataset import DatasetId, get_dataloaders
from neurotorch import Dimension, DimensionProperty
from neurotorch.callbacks import CheckpointManager, LoadCheckpointMode
from neurotorch.callbacks.training_visualization import TrainingHistoryVisualizationCallback
from neurotorch.metrics import ClassificationMetrics
from neurotorch.modules import SequentialModel, ALIFLayer, LILayer, SpikeFuncType, SpikeFuncType2Func
from neurotorch.modules.layers import LayerType, LayerType2Layer, LearningType
from neurotorch.trainers import ClassificationTrainer
from neurotorch.transforms import ConstantValuesTransform, LinearRateToSpikes
from neurotorch.transforms.vision import ImgToSpikes
from neurotorch.utils import hash_params


def get_training_params_space() -> Dict[str, Any]:
	"""
	Get the parameters space for the training.
	:return: The parameters space.
	"""
	return {
		"dataset_id": [
			DatasetId.MNIST,
			# DatasetId.FASHION_MNIST
		],
		"input_transform": [
			# "linear",
			"NorseConstCurrLIF",
			# "ImgToSpikes",
		],
		"n_steps": [
			# 2,
			10,
			32,
			100,
			# 1_000
		],
		"n_hidden_neurons": [
			# 16,
			# 32,
			# 64,
			# [64, 64],
			128,
			# 256,
			# [32, 32],
			# 32
		],
		# "spike_func": [SpikeFuncType.FastSigmoid, ],
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
		# 	# True,
		# 	False
		# ],
	}


def get_meta_name(params: Dict[str, Any]):
	meta_name = f""
	for k, v in params.items():
		meta_name += f"{k}-{v}_"
	return meta_name[:-1]


def save_params(params: Dict[str, Any], save_path: str):
	"""
	Save the parameters in a file.
	:param save_path: The path to save the parameters.
	:param params: The parameters to save.
	:return: The path to the saved parameters.
	"""
	pickle.dump(params, open(save_path, "wb"))
	return save_path


def get_transform_from_str(transform_name: str, **kwargs):
	kwargs.setdefault("dt", 1e-3)
	kwargs.setdefault("n_steps", 10)
	name_to_transform = {
		"none": None,
		"linear": Compose([torch.flatten, LinearRateToSpikes(n_steps=kwargs["n_steps"])]),
		"NorseConstCurrLIF": Compose([
			torch.flatten, norse.torch.ConstantCurrentLIFEncoder(seq_length=kwargs["n_steps"], dt=kwargs["dt"])
		]),
		"ImgToSpikes": Compose([torch.flatten, ImgToSpikes(n_steps=kwargs["n_steps"], use_periods=True)]),
		"flatten": Compose([torch.flatten, Lambda(lambda x: x[np.newaxis, :])]),
		"const": Compose([torch.flatten, ConstantValuesTransform(n_steps=kwargs["n_steps"])]),
	}
	name_to_transform = {k.lower(): v for k, v in name_to_transform.items()}
	return name_to_transform[transform_name.lower()]


def train_with_params(
		params: Dict[str, Any],
		n_iterations: int = 100,
		batch_size: int = 256,
		data_folder: str = "tr_results",
		verbose: bool = False,
		show_training: bool = False,
		force_overwrite: bool = False,
):
	checkpoints_name = str(hash_params(params))
	checkpoint_folder = f"{data_folder}/{checkpoints_name}"
	os.makedirs(checkpoint_folder, exist_ok=True)
	if verbose:
		logging.info(f"Checkpoint folder: {checkpoint_folder}")

	n_features = 28 * 28

	dataloaders = get_dataloaders(
		dataset_id=params["dataset_id"],
		batch_size=batch_size,
		input_transform=get_transform_from_str(params["input_transform"], **params),
		train_val_split_ratio=params.get("train_val_split_ratio", 0.85),
		nb_workers=psutil.cpu_count(logical=False),
	)
	n_hidden_neurons = params["n_hidden_neurons"]
	if not isinstance(n_hidden_neurons, Iterable):
		n_hidden_neurons = [n_hidden_neurons]
	n_hidden_neurons.insert(0, n_features)

	hidden_layers = [
		LayerType2Layer[params["hidden_layer_type"]](
			input_size=n_hidden_neurons[i],
			output_size=n,
			# spike_func=SpikeFuncType2Func[params["spike_func"]],
			**params
		)
		for i, n in enumerate(n_hidden_neurons[1:])
	] if len(n_hidden_neurons) > 1 else []
	input_params = deepcopy(params)
	input_params.pop("forward_weights", None)
	input_params.pop("use_recurrent_connection", None)
	input_params.pop("learning_type", None)
	network = SequentialModel(
		layers=[
			LayerType2Layer[params["input_layer_type"]](
				input_size=Dimension(n_features, DimensionProperty.NONE),
				# spike_func=SpikeFuncType2Func[params["spike_func"]],
				output_size=n_hidden_neurons[0],
				forward_weights=torch.eye(n_features),
				use_recurrent_connection=False,
				learning_type=input_params.get("input_learning_type", LearningType.NONE),
				**input_params
			),
			*hidden_layers,
			LayerType2Layer[params["readout_layer_type"]](output_size=10),
		],
		name=f"{params['dataset_id'].name}_network_{checkpoints_name}",
		checkpoint_folder=checkpoint_folder,
		foresight_time_steps=params.get("foresight_time_steps", 0),
	)
	network.build()
	if verbose:
		logging.info(f"\nNetwork:\n{network}")
	checkpoint_manager = CheckpointManager(checkpoint_folder)
	save_params(params, os.path.join(checkpoint_folder, "params.pkl"))
	callbacks = [checkpoint_manager, ]
	if show_training:
		callbacks.append(TrainingHistoryVisualizationCallback("./temp/"))
	trainer = ClassificationTrainer(
		model=network,
		callbacks=callbacks,
		verbose=verbose,
	)
	trainer.train(
		dataloaders["train"],
		dataloaders["val"],
		n_iterations=n_iterations,
		load_checkpoint_mode=LoadCheckpointMode.LAST_ITR if not force_overwrite else None,
		force_overwrite=force_overwrite,
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


def get_all_params_combinations(params_space: Dict[str, Any] = None) -> List[Dict[str, Any]]:
	if params_space is None:
		params_space = get_training_params_space()
	# get all the combinaison of the parameters
	all_params = list(params_space.keys())
	all_params_values = list(params_space.values())
	all_params_combinaison = list(map(lambda x: list(x), list(itertools.product(*all_params_values))))

	# create a list of dict of all the combinaison
	all_params_combinaison_dict = list(map(lambda x: dict(zip(all_params, x)), all_params_combinaison))
	return all_params_combinaison_dict


def train_all_params(
		training_params: Dict[str, Any] = None,
		n_iterations: int = 100,
		batch_size: int = 256,
		data_folder: str = "tr_results",
		verbose: bool = False,
		rm_data_folder_and_restart_all_training: bool = False,
		force_overwrite: bool = False,
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
			if str(hash_params(params)) in df["checkpoints"].values:
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



