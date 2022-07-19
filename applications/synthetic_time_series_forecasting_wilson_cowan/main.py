import os
import pprint
from collections import OrderedDict
from typing import Any, Dict

import numpy as np
import torch.onnx
import torch
from matplotlib import pyplot as plt

from dataset import WilsonCowanTimeSeries, get_dataloaders
from neurotorch.callbacks import CheckpointManager, LoadCheckpointMode
from neurotorch.metrics import RegressionMetrics
from neurotorch.modules import SequentialModel
from neurotorch.modules.layers import WilsonCowanLayer
from neurotorch.trainers import RegressionTrainer
from neurotorch.utils import hash_params
from neurotorch.visualisation.time_series_visualisation import Visualise, VisualiseKMeans

# TODO : calculate pVar on each neurons instead of calculating it on batch wise
# TODO : Problem when ratio != 0.5


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
			std_weight=params["std_weight"],
			learn_r=params["learn_r"],
			std_r=params["std_r"],
		)
		for _ in range(params["num_hidden_layers"])
	]
	network = SequentialModel(
		layers=hidden_layer,
		checkpoint_folder=checkpoint_folder,
		device=torch.device("cuda"),
		foresight_time_steps=int(params["chunk_size"] * params["ratio"])
	)
	network.build()
	checkpoint_manager = CheckpointManager(checkpoint_folder, minimise_metric=False)
	trainer = RegressionTrainer(
		model=network,
		callbacks=checkpoint_manager,
		criterion=lambda x, y: RegressionMetrics.compute_p_var(y, x),
		optimizer=torch.optim.Adam(network.parameters(), lr=1e-3, maximize=True),
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
	n_neurons = 2  # Num of neurons
	num_step = 100
	dt = 0.1
	t_0 = np.random.rand(n_neurons, )
	forward_weights = 2 * np.random.randn(n_neurons, n_neurons)
	mu = 0
	r = 0
	tau = 1
	WCTS = WilsonCowanTimeSeries(num_step, dt, t_0, forward_weights, mu, r, tau)
	time_series = WCTS.compute_timeseries()

	results = train_with_params(
		{
			"time_series": time_series,
			"batch_size": 1,
			"train_val_split_ratio": 0.8,
			"chunk_size": 50,
			"ratio": 0.5,
			"num_hidden_layers": 1,
			"dt": dt,
			"std_weight": 1.2,
			"learn_r": False,
			"std_r": 0,
			"hidden_layer_size": n_neurons,
		},
		n_iterations=50,
		verbose=True
	)
	pprint.pprint(results, indent=4)
	# WCTS.plot_timeseries(False)

	pred_forward_weights = list(results['network'].output_layers.values())[0].forward_weights.detach().cpu().numpy()
	pred_WCTS = WilsonCowanTimeSeries(num_step, dt, t_0, pred_forward_weights, mu, r, tau)
	pred_time_series = pred_WCTS.compute_timeseries()
	#VisualiseKMeans(time_series).heatmap()
	#VisualiseKMeans(pred_time_series).heatmap()
	for idx in range(n_neurons):
		plt.plot(time_series[idx], label='true')
		plt.plot(pred_time_series[idx], label='pred')
		plt.legend()
		plt.show()

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

