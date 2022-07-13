import itertools
import logging
import os
import pickle
import warnings
from collections import OrderedDict
from typing import Any, Dict, Iterable, List

import pandas as pd
import psutil
import tqdm

from applications.mnist.dataset import DatasetId, get_dataloaders
from neurotorch import Dimension, DimensionProperty
from neurotorch.callbacks import CheckpointManager, LoadCheckpointMode
from neurotorch.metrics import ClassificationMetrics
from neurotorch.modules import SequentialModel, ALIFLayer, LILayer, SpikeFuncType, SpikeFuncType2Func
from neurotorch.modules.layers import LayerType, LayerType2Layer, LearningType
from neurotorch.trainers import ClassificationTrainer
from neurotorch.utils import hash_params


def get_training_params_space() -> Dict[str, Any]:
	"""
	Get the parameters space for the training.
	:return: The parameters space.
	"""
	return {
		"dataset_id": [
			DatasetId.MNIST,
			DatasetId.FASHION_MNIST
		],
		"to_spikes_use_periods": [
			True,
			# False,
		],
		"inputs_linear": [
			True,
			# False,
		],
		"n_steps": [
			2,
			10,
			100,
			# 1_000
		],
		"n_hidden_neurons": [
			# 16,
			# 32,
			# 64,
			# [64, 64],
			128,
			# [32, 32],
			# 32
		],
		"spike_func": [SpikeFuncType.FastSigmoid, ],
		"hidden_layer_type": [
			LayerType.LIF,
			LayerType.ALIF,
		],
		"use_recurrent_connection": [
			False,
			True
		],
		"learn_beta": [
			True,
			# False
		],
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


def train_with_params(params: Dict[str, Any], n_iterations: int = 100, data_folder="tr_results", verbose=False):
	checkpoints_name = str(hash_params(params))
	checkpoint_folder = f"{data_folder}/{checkpoints_name}"
	os.makedirs(checkpoint_folder, exist_ok=True)
	if verbose:
		print(f"Checkpoint folder: {checkpoint_folder}")

	dataloaders = get_dataloaders(
		dataset_id=params["dataset_id"],
		batch_size=256,
		as_timeseries=True,
		inputs_linear=params["inputs_linear"],
		n_steps=params["n_steps"],
		to_spikes_use_periods=params["to_spikes_use_periods"],
		train_val_split_ratio=params.get("train_val_split_ratio", 0.85),
		nb_workers=psutil.cpu_count(logical=False),
	)
	n_hidden_neurons = params["n_hidden_neurons"]
	if not isinstance(n_hidden_neurons, Iterable):
		n_hidden_neurons = [n_hidden_neurons]

	hidden_layers = [
		LayerType2Layer[params["hidden_layer_type"]](
			use_recurrent_connection=params["use_recurrent_connection"],
			learn_beta=params["learn_beta"],
			input_size=n_hidden_neurons[i],
			output_size=n,
			spike_func=SpikeFuncType2Func[params["spike_func"]],
		)
		for i, n in enumerate(n_hidden_neurons[1:])
	] if len(n_hidden_neurons) > 1 else []
	network = SequentialModel(
		layers=[
			LayerType2Layer[params["hidden_layer_type"]](
				input_size=Dimension(28*28, DimensionProperty.NONE),
				use_recurrent_connection=params["use_recurrent_connection"],
				learn_beta=params["learn_beta"],
				spike_func=SpikeFuncType2Func[params["spike_func"]],
				output_size=n_hidden_neurons[0],
			),
			*hidden_layers,
			LILayer(output_size=10),
		],
		name=f"mnist_network_{checkpoints_name}",
		checkpoint_folder=checkpoint_folder,
	)
	network.build()
	checkpoint_manager = CheckpointManager(checkpoint_folder)
	save_params(params, os.path.join(checkpoint_folder, "params.pkl"))
	trainer = ClassificationTrainer(
		model=network,
		callbacks=checkpoint_manager,
		verbose=verbose,
	)
	trainer.train(
		dataloaders["train"],
		dataloaders["val"],
		n_iterations=n_iterations,
		load_checkpoint_mode=LoadCheckpointMode.LAST_ITR,
		# force_overwrite=True,
	)
	try:
		network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.BEST_ITR, verbose=verbose)
	except FileNotFoundError:
		if verbose:
			print("No best checkpoint found. Loading last checkpoint instead.")
		network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.LAST_ITR, verbose=verbose)
	return OrderedDict(dict(
		network=network,
		checkpoints_name=checkpoints_name,
		accuracies={
			k: ClassificationMetrics.accuracy(network, dataloaders[k], verbose=verbose, desc=f"{k}_accuracy")
			for k in dataloaders
		},
		precisions={
			k: ClassificationMetrics.precision(network, dataloaders[k], verbose=verbose, desc=f"{k}_precision")
			for k in dataloaders
		},
		recalls={
			k: ClassificationMetrics.recall(network, dataloaders[k], verbose=verbose, desc=f"{k}_recall")
			for k in dataloaders
		},
		f1s={
			k: ClassificationMetrics.f1(network, dataloaders[k], verbose=verbose, desc=f"{k}_f1")
			for k in dataloaders
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
		data_folder: str = "tr_results",
		verbose=False
):
	"""
	Train the network with all the parameters.
	:param n_iterations: The number of iterations to train the network.
	:param verbose: If True, print the progress.
	:param data_folder: The folder where to save the data.
	:param training_params: The parameters to use for the training.
	:return: The results of the training.
	"""
	warnings.filterwarnings("ignore", category=UserWarning)
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
				result = train_with_params(params, n_iterations=n_iterations, data_folder=data_folder, verbose=verbose)
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



