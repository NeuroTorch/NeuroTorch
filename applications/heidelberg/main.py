
import os
import pprint
import multiprocessing as mp
import warnings
from threading import Thread
from typing import Any, Dict
from collections import OrderedDict

import matplotlib.pyplot as plt
import psutil
import torch

from applications.heidelberg.dataset import HeidelbergDataset, get_dataloaders
from neurotorch import Dimension, DimensionProperty
from neurotorch.callbacks import CheckpointManager, LoadCheckpointMode
from neurotorch.callbacks.base_callback import BaseCallback
from neurotorch.callbacks.training_visualization import TrainingHistoryVisualizationCallback
from neurotorch.metrics import ClassificationMetrics
from neurotorch.modules import SequentialModel, ALIFLayer, LILayer
from neurotorch.modules.layers import LIFLayer, LearningType, SpyLIFLayer, SpyLILayer
from neurotorch.trainers import ClassificationTrainer, Trainer
from neurotorch.utils import hash_params


def train_with_params(
		params: Dict[str, Any],
		n_iterations: int = 100,
		batch_size: int = 256,
		data_folder="tr_results",
		verbose=False
):
	checkpoints_name = str(hash_params(params))
	checkpoint_folder = f"{data_folder}/{checkpoints_name}"
	os.makedirs(checkpoint_folder, exist_ok=True)
	print(f"Checkpoint folder: {checkpoint_folder}")

	dataloaders = get_dataloaders(
		batch_size=batch_size,
		n_steps=params["n_steps"],
		as_sparse=True,
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
			SpyLIFLayer(
				input_size=[
					Dimension(None, DimensionProperty.TIME),
					Dimension(dataloaders["test"].dataset.n_units, DimensionProperty.NONE)
				],
				use_recurrent_connection=params["use_recurrent_connection"],
				# learn_beta=params["learn_beta"],
				# use_rec_eye_mask=params["use_rec_eye_mask"],
				output_size=[
					Dimension(None, DimensionProperty.TIME),
					Dimension(params.get("n_hidden_neurons"), DimensionProperty.NONE)
				],
			),
			*hidden_layers,
			SpyLILayer(output_size=dataloaders["test"].dataset.n_classes, use_bias=False),
		],
		name="heidelberg_network",
		checkpoint_folder=checkpoint_folder,
	)
	network.build()
	checkpoint_manager = CheckpointManager(checkpoint_folder)
	# save_params(params, os.path.join(checkpoint_folder, "params.pkl"))
	trainer = ClassificationTrainer(
		model=network,
		callbacks=[checkpoint_manager, TrainingHistoryVisualizationCallback("./temp/")],
		optimizer=torch.optim.Adamax(network.parameters(), lr=2e-4),
	)
	training_history = trainer.train(
		dataloaders["train"],
		dataloaders["val"],
		n_iterations=n_iterations,
		load_checkpoint_mode=LoadCheckpointMode.LAST_ITR,
		# force_overwrite=True,
		verbose=verbose,
	)
	training_history.plot(
		save_path=os.path.join(checkpoint_folder, "training_history.png"),
		show=False,
		close=True,
	)
	try:
		network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.BEST_ITR, verbose=verbose)
	except FileNotFoundError:
		warnings.warn("No best checkpoint found. Loading last checkpoint instead.", RuntimeWarning)
		network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.LAST_ITR, verbose=verbose)
	return OrderedDict(dict(
		network=network,
		checkpoints_name=checkpoints_name,
		training_history=training_history,
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
	torch.manual_seed(42)
	results = train_with_params(
		{
			"use_recurrent_connection": True,
			"n_hidden_layers": 0,
			"n_hidden_neurons": 512,
			# "learn_beta": False,
			# "use_rec_eye_mask": False,
			"n_steps": 100,
			"train_val_split_ratio": 0.95,
		},
		n_iterations=500,
		batch_size=1024,
		verbose=True,
	)
	pprint.pprint(results, indent=4)
	results["training_history"].plot(show=True)


