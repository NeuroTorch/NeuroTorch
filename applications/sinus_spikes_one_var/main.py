import os
import pprint
from typing import Any, Dict
from collections import OrderedDict

import psutil

from applications.sinus_spikes_one_var.dataset import SinusSpikesDataset, get_dataloaders
from neurotorch import Dimension, DimensionProperty
from neurotorch.callbacks import CheckpointManager, LoadCheckpointMode
from neurotorch.metrics import RegressionMetrics
from neurotorch.modules import SequentialModel, ALIFLayer, LILayer
from neurotorch.trainers import RegressionTrainer
from neurotorch.utils import hash_params


def train_with_params(params: Dict[str, Any], n_iterations: int = 100, data_folder="tr_results", verbose=False):
	checkpoints_name = str(hash_params(params))
	checkpoint_folder = f"{data_folder}/{checkpoints_name}"
	os.makedirs(checkpoint_folder, exist_ok=True)
	print(f"Checkpoint folder: {checkpoint_folder}")
	n_variables = params["n_variables"]
	n_steps = params["n_steps"]
	dataloaders = get_dataloaders(
		batch_size=256,
		n_steps=n_steps,
		n_variables=n_variables,
		train_val_split_ratio=params.get("train_val_split_ratio", 0.85),
		nb_workers=psutil.cpu_count(logical=False),
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
			ALIFLayer(
				input_size=Dimension(n_variables, DimensionProperty.NONE),
				use_recurrent_connection=params["use_recurrent_connection"],
				learn_beta=params["learn_beta"],
			),
			*hidden_layers,
			ALIFLayer(
				use_recurrent_connection=params["use_recurrent_connection"],
				learn_beta=params["learn_beta"],
				output_size=n_variables
			),
		],
		foresight_time_steps=n_steps,
		name="mnist_network",
		checkpoint_folder=checkpoint_folder,
	)
	network.build()
	checkpoint_manager = CheckpointManager(checkpoint_folder)
	# save_params(params, os.path.join(checkpoint_folder, "params.pkl"))
	trainer = RegressionTrainer(
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
		network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.LAST_ITR, verbose=verbose)
	return OrderedDict(dict(
		network=network,
		checkpoints_name=checkpoints_name,
		mae={
			k: RegressionMetrics.mean_absolute_error(network, dataloaders[k], verbose=True, desc=f"{k}_mae")
			for k in dataloaders
		},
		mse={
			k: RegressionMetrics.mean_squared_error(network, dataloaders[k], verbose=True, desc=f"{k}_mse")
			for k in dataloaders
		},
		r2={
			k: RegressionMetrics.r2(network, dataloaders[k], verbose=True, desc=f"{k}_r2")
			for k in dataloaders
		},
		d2={
			k: RegressionMetrics.d2_tweedie(network, dataloaders[k], verbose=True, desc=f"{k}_d2")
			for k in dataloaders
		},
	))


if __name__ == '__main__':
	SinusSpikesDataset(n_variables=25, noise_std=0.1, n_steps=10, n_samples=1).show()
	results = train_with_params(
		{
			"to_spikes_use_periods": False,
			"use_recurrent_connection": False,
			"n_hidden_layers": 0,
			"n_hidden_neurons": 128,
			"learn_beta": False,
			"n_steps": 10,
			"train_val_split_ratio": 0.95,
		},
		n_iterations=100,
		verbose=True,
	)
	pprint.pprint(results, indent=4)
