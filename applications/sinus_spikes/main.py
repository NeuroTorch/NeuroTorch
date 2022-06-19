import os
import pprint
from typing import Any, Dict
from collections import OrderedDict

import numpy as np
import psutil
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from applications.sinus_spikes.dataset import SinusSpikesDataset, get_dataloaders
from neurotorch import Dimension, DimensionProperty
from neurotorch.callbacks import CheckpointManager, LoadCheckpointMode
from neurotorch.metrics import RegressionMetrics
from neurotorch.modules import SequentialModel, ALIFLayer, LILayer
from neurotorch.trainers import RegressionTrainer
from neurotorch.utils import hash_params


def show_prediction(dataloader: DataLoader, network: SequentialModel):
	y_prev, y_next = dataloader.dataset[0]
	y_pred = network.get_prediction_trace(y_prev.unsqueeze(0)).squeeze()
	y_prev, y_next, y_pred = y_prev.detach().cpu().numpy(), y_next.detach().cpu().numpy(), y_pred.detach().cpu().numpy()

	t_prev = np.arange(y_prev.shape[0])
	t_next = np.arange(y_next.shape[0])
	fig, ax = plt.subplots(figsize=(10, 5))
	line_length = 1.0
	pad = 0.5
	for n_idx, spikes in enumerate(y_prev.T):
		spikes_idx = t_prev[np.isclose(spikes, 1.0)]
		ymin = (spikes.shape[-1] - n_idx) * (pad + line_length)
		ax.vlines(spikes_idx, ymin=ymin, ymax=ymin + line_length, colors=[0, 0, 0])
	for n_idx, (spikes, spikes_pred) in enumerate(zip(y_next.T, y_pred.T)):
		spikes_idx = t_next[np.isclose(spikes, 1.0)].astype(int)
		spikes_pred_idx = t_next[np.isclose(spikes_pred, 1.0)].astype(int)
		ymin = (spikes.shape[-1] - n_idx) * (pad + line_length)
		ax.vlines(spikes_idx, ymin=ymin, ymax=ymin + line_length, colors=[0, 0, 0])
		good_pred = np.asarray([s in spikes_idx for s in spikes_pred_idx])
		good_pred_color = 'g'
		bad_pred_color = 'r'
		if len(good_pred) > 0:
			colors = np.where(good_pred, good_pred_color, bad_pred_color)
		else:
			colors = bad_pred_color
		ax.vlines(spikes_idx+y_prev.shape[0], ymin=ymin, ymax=ymin + line_length, colors=[0, 0, 0])
		ax.vlines(spikes_pred_idx+y_prev.shape[0], ymin=ymin, ymax=ymin + line_length, colors=colors)
	ax.get_yaxis().set_visible(False)
	plt.show()


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
	show_prediction(dataloaders["test"], network)
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
	results = train_with_params(
		{
			"use_recurrent_connection": True,
			"n_hidden_layers": 1,
			"n_hidden_neurons": 128,
			"learn_beta": True,
			"n_steps": 100,
			"n_variables": 30,
			"train_val_split_ratio": 0.95,
		},
		n_iterations=100,
		verbose=True,
	)
	pprint.pprint(results, indent=4)
