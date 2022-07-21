import os
import pprint
from collections import OrderedDict
from typing import Any, Dict

import numpy as np
import torch.onnx
import torch
from matplotlib import pyplot as plt

from dataset import WilsonCowanTimeSeries, get_dataloaders, get_dataloaders_CURBD
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
	# checkpoints_name = str(hash_params(params))
	checkpoints_name = "curbd_trial"
	checkpoint_folder = f"{data_folder}/{checkpoints_name}"
	os.makedirs(checkpoint_folder, exist_ok=True)
	print(f"Checkpoint folder: {checkpoint_folder}")
	dataloaders = get_dataloaders_CURBD(
		time_series=params["time_series"],
		# batch_size=params["batch_size"],
		# train_val_split_ratio=params["train_val_split_ratio"],
		# chunk_size=params["chunk_size"],
		# ratio=params["ratio"],
		# nb_workers=psutil.cpu_count(logical=False)
	)
	hidden_layer = [
		WilsonCowanLayer(
			input_size=params["hidden_layer_size"],
			output_size=params["hidden_layer_size"],
			dt=params["dt"],
			std_weight=params["std_weight"],
			learn_r=params["learn_r"],
			learn_mu=params["learn_mu"],
			std_r=params["std_r"],
		)
		for _ in range(params["num_hidden_layers"])
	]
	network = SequentialModel(
		layers=hidden_layer,
		checkpoint_folder=checkpoint_folder,
		device=torch.device("cuda"),
		# foresight_time_steps=round(params["chunk_size"] * (1-params["ratio"]))
		foresight_time_steps=params["num_step"]-1,
	)
	network.build()
	#checkpoint_manager = CheckpointManager(checkpoint_folder, minimise_metric=True)
	checkpoint_manager = CheckpointManager(checkpoint_folder, minimise_metric=False)
	reg_metrics = RegressionMetrics(network, metrics_names=["mse", "p_var"])
	trainer = RegressionTrainer(
		model=network,
		callbacks=checkpoint_manager,
		criterion=lambda pred, y: RegressionMetrics.compute_p_var(y_true=y, y_pred=pred, reduction='mean'),
		optimizer=torch.optim.Adam(network.parameters(), lr=1e-2, maximize=True),
		#optimizer=torch.optim.Adam(network.parameters(), lr=1e-2),
		metrics=[reg_metrics]
	)
	history = trainer.train(
		dataloaders["train"],
		dataloaders["val"],
		n_iterations=n_iterations,
		#load_checkpoint_mode=LoadCheckpointMode.LAST_ITR,
		force_overwrite=True,
		verbose=verbose
	)
	# history.plot(show=True)
	try:
		network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.LAST_ITR, verbose=verbose)
	except FileNotFoundError:
		network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.LAST_ITR, verbose=verbose)
	# show_prediction(dataloaders["test"], network)  # TODO: not implemented yet
	return OrderedDict(dict(
		network=network,
		checkpoints_name=checkpoints_name,
		# mae={
		# 	k: RegressionMetrics.mean_absolute_error(network, dataloaders[k], verbose=True, desc=f"{k}_mae")
		# 	for k in dataloaders
		# },
		# mse={
		# 	k: RegressionMetrics.mean_squared_error(network, dataloaders[k], verbose=True, desc=f"{k}_mse")
		# 	for k in dataloaders
		# },
		# r2={
		# 	k: RegressionMetrics.r2(network, dataloaders[k], verbose=True, desc=f"{k}_r2")
		# 	for k in dataloaders
		# },
		# d2={
		# 	k: RegressionMetrics.d2_tweedie(network, dataloaders[k], verbose=True, desc=f"{k}_d2")
		# 	for k in dataloaders
		# },
		# p_var={
		# 	k: RegressionMetrics.p_var(network, dataloaders[k], verbose=True, desc=f"{k}_pVar")
		# 	for k in dataloaders
		# },
	))


if __name__ == '__main__':
	n_neurons = 20  # Num of neurons
	num_step = 150
	dt = 0.1
	t_0 = np.random.rand(n_neurons, )
	forward_weights = 2 * np.random.randn(n_neurons, n_neurons)
	mu = 0
	r = 0
	tau = 1
	WCTS = WilsonCowanTimeSeries(num_step, dt, t_0, forward_weights, mu, r, tau)
	time_series = WCTS.compute()
	Visualise(time_series).plot_timeseries()



	results = train_with_params(
		{
			"time_series": time_series,
			"batch_size": 1,
			"train_val_split_ratio": 0.9,
			"num_step": num_step,
			"chunk_size": 150,
			"ratio": 0.5,
			"num_hidden_layers": 1,
			"dt": dt,
			"std_weight": 2,
			"learn_r": False,
			"learn_mu": False,
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
	pred_seq = results['network'].get_prediction_trace(torch.from_numpy(t_0[np.newaxis, np.newaxis, :])).detach().cpu().numpy()
	pred_seq = np.transpose(pred_seq[0])
	pred_seq = np.concatenate([t_0[:, np.newaxis], pred_seq], axis=1)
	print(f"{pred_seq.shape = }")
	pred_time_series = pred_WCTS.compute()
	#VisualiseKMeans(time_series).heatmap()
	#VisualiseKMeans(pred_time_series).heatmap()
	p_var = RegressionMetrics.compute_p_var(
		time_series[np.newaxis, :],
		pred_time_series[np.newaxis, :],
		reduction='mean'
	)
	print(f"{p_var = }")
	for idx in range(n_neurons):
		plt.plot(time_series[idx], label='true')
		plt.plot(pred_time_series[idx], label='pred')
		plt.plot(pred_seq[idx], label='pred_seq')
		plt.legend()
		plt.ylim([0, 1])
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

