
import os
import pprint
from typing import Any, Dict
from collections import OrderedDict

import psutil

from applications.heidelberg.dataset import HeidelbergDataset, get_dataloaders
from neurotorch import Dimension, DimensionProperty
from neurotorch.callbacks import CheckpointManager, LoadCheckpointMode
from neurotorch.metrics import ClassificationMetrics
from neurotorch.modules import SequentialModel, ALIFLayer, LILayer
from neurotorch.modules.layers import LIFLayer, LearningType
from neurotorch.trainers import ClassificationTrainer
from neurotorch.utils import hash_params


def train_with_params(params: Dict[str, Any], n_iterations: int = 100, data_folder="tr_results", verbose=False):
	checkpoints_name = str(hash_params(params))
	checkpoint_folder = f"{data_folder}/{checkpoints_name}"
	os.makedirs(checkpoint_folder, exist_ok=True)
	print(f"Checkpoint folder: {checkpoint_folder}")

	dataloaders = get_dataloaders(
		batch_size=256,
		n_steps=params["n_steps"],
		train_val_split_ratio=params.get("train_val_split_ratio", 0.85),
		# nb_workers=psutil.cpu_count(logical=False),
	)
	hidden_layers = [
		ALIFLayer(
			use_recurrent_connection=params["use_recurrent_connection"],
			learn_beta=params["learn_beta"],
		)
		for _ in range(params["n_hidden_layers"])
	]
	network = SequentialModel(
		layers=[
			LIFLayer(
				input_size=[
					Dimension(None, DimensionProperty.TIME),
					Dimension(dataloaders["test"].dataset.n_units, DimensionProperty.NONE)
				],
				use_recurrent_connection=params["use_recurrent_connection"],
				# learn_beta=params["learn_beta"],
			),
			*hidden_layers,
			LILayer(output_size=dataloaders["test"].dataset.n_classes),
		],
		name="heidelberg_network",
		checkpoint_folder=checkpoint_folder,
	)
	network.build()
	checkpoint_manager = CheckpointManager(checkpoint_folder)
	# save_params(params, os.path.join(checkpoint_folder, "params.pkl"))
	trainer = ClassificationTrainer(
		model=network,
		callbacks=checkpoint_manager,
	)
	trainer.train(
		dataloaders["train"],
		dataloaders["val"],
		n_iterations=n_iterations,
		load_checkpoint_mode=LoadCheckpointMode.LAST_ITR,
		# force_overwrite=True,
		verbose=verbose,
	)
	try:
		network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.BEST_ITR, verbose=verbose)
	except FileNotFoundError:
		print("No best checkpoint found. Loading last checkpoint instead.")
		network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.LAST_ITR, verbose=verbose)
	return OrderedDict(dict(
		network=network,
		checkpoints_name=checkpoints_name,
		accuracies={
			k: ClassificationMetrics.accuracy(network, dataloaders[k], verbose=True, desc=f"{k}_accuracy")
			for k in dataloaders
		},
		precisions={
			k: ClassificationMetrics.precision(network, dataloaders[k], verbose=True, desc=f"{k}_precision")
			for k in dataloaders
		},
		recalls={
			k: ClassificationMetrics.recall(network, dataloaders[k], verbose=True, desc=f"{k}_recall")
			for k in dataloaders
		},
		f1s={
			k: ClassificationMetrics.f1(network, dataloaders[k], verbose=True, desc=f"{k}_f1")
			for k in dataloaders
		},
	))


if __name__ == '__main__':
	results = train_with_params(
		{
			"use_recurrent_connection": False,
			"n_hidden_layers": 0,
			"n_hidden_neurons": 128,
			# "learn_beta": True,
			"n_steps": 100,
			"train_val_split_ratio": 0.95,
		},
		n_iterations=200,
		verbose=True,
	)
	pprint.pprint(results, indent=4)


