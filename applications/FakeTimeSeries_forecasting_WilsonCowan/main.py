import os
import pprint
from collections import OrderedDict
from typing import Any, Dict

import numpy as np
import torch.onnx
from matplotlib import pyplot as plt

from dataset import WilsonCowanTimeSeries, get_dataloaders
from neurotorch.callbacks import CheckpointManager, LoadCheckpointMode
from neurotorch.metrics import RegressionMetrics
from neurotorch.modules import SequentialModel
from neurotorch.modules.layers import WilsonCowanLayer
from neurotorch.trainers import RegressionTrainer
from neurotorch.utils import hash_params


def p_var(output, target):
	loss = 1 - (torch.norm(output - target) / torch.norm(target - torch.mean(target)))**2
	return loss


def train_with_params(params: Dict[str, Any], n_iterations: int = 100, data_folder="tr_results", verbose=True):
	checkpoints_name = str(hash_params(params))
	checkpoint_folder = f"{data_folder}/{checkpoints_name}"
	os.makedirs(checkpoint_folder, exist_ok=True)
	print(f"Checkpoint folder: {checkpoint_folder}")
	dataloaders = get_dataloaders(
		time_series=params["time_series"],
		batch_size=params["batch_size"],
		train_val_split_ratio=params["train_val_split_ratio"],
		chunk_size=params["chunk_size"],
		ratio=params["ratio"],
		# nb_workers=psutil.cpu_count(logical=False)
	)
	hidden_layer = [
		WilsonCowanLayer(
			input_size=params["hidden_layer_size"],
			output_size=params["hidden_layer_size"],
			dt=params["dt"],
			device=torch.device("cpu"),
		)
		for _ in range(params["num_hidden_layers"])
	]
	network = SequentialModel(
		layers=hidden_layer,
		checkpoint_folder=checkpoint_folder,
		device=torch.device("cpu"),
	)
	network.build()
	checkpoint_manager = CheckpointManager(checkpoint_folder, minimise_metric=False)
	trainer = RegressionTrainer(
		model=network,
		callbacks=checkpoint_manager,
		criterion=p_var,
		optimizer=torch.optim.Adam(network.parameters(), lr=1e-3, maximize=True),
		device=torch.device("cpu"),
	)
	trainer.train(
		dataloaders["train"],
		dataloaders["val"],
		n_iterations=n_iterations,
		load_checkpoint_mode=LoadCheckpointMode.LAST_ITR,
		# force_overwrite=True,
		verbose=verbose
	)
	try:
		network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.BEST_ITR, verbose=verbose)
	except FileNotFoundError:
		network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.LAST_ITR, verbose=verbose)
	# show_prediction(dataloaders["test"], network)  # TODO: not implemented yet
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
		p_var={
			k: RegressionMetrics.p_var(network, dataloaders[k], verbose=True, desc=f"{k}_pVar")
			for k in dataloaders
		},
	))


if __name__ == '__main__':
	n_neurons = 128  # Num of neurons
	num_step = 1_000
	dt = 0.1
	t_0 = np.random.rand(n_neurons, )
	forward_weights = 8 * np.random.randn(n_neurons, n_neurons)
	mu = 0
	r = np.random.rand(n_neurons, ) * 2
	tau = 1
	WCTS = WilsonCowanTimeSeries(num_step, dt, t_0, forward_weights, mu, r, tau)
	time_series = WCTS.compute_timeseries()

	results = train_with_params(
		{
			"time_series": time_series,
			"batch_size": 1024,
			"train_val_split_ratio": 0.8,
			"chunk_size": 100,
			"ratio": 0.5,
			"num_hidden_layers": 1,
			"dt": dt,
			"hidden_layer_size": n_neurons,
		},
		n_iterations=100,
		verbose=True
	)
	pprint.pprint(results, indent=4)
	# WCTS.plot_timeseries(False)

	pred_forward_weights = list(results['network'].output_layers.values())[0].forward_weights.detach().cpu().numpy()
	pred_WCTS = WilsonCowanTimeSeries(num_step, dt, t_0, pred_forward_weights, mu, r, tau)
	pred_time_series = pred_WCTS.compute_timeseries()
	# pred_WCTS.plot_timeseries(False)

	# for i in range(n_neurons):
	# 	time = np.linspace(0, pred_WCTS.num_step * pred_WCTS.dt, pred_WCTS.num_step)
	# 	plt.plot(time.T, pred_time_series[i].T, "r", label="pred")
	# 	plt.plot(time.T, time_series[i].T, "b", label="true")
	# 	plt.xlabel('Time')
	# 	plt.ylabel(f'Neuronal activity {i}')
	# 	plt.ylim([0, 1])
	# 	plt.legend()
	# 	plt.show()
	# print(f"{forward_weights = }")
	# print(f"{pred_forward_weights = }")

