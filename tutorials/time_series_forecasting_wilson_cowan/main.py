from typing import Union, Iterable, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader

import neurotorch as nt
from dataset import WSDataset
from neurotorch.callbacks.convergence import ConvergenceTimeGetter
from neurotorch.callbacks.early_stopping import EarlyStoppingThreshold
from neurotorch.modules.functions import WeirdTanh
from neurotorch.modules.layers import WilsonCowanLayer
from neurotorch.metrics import RegressionMetrics
from neurotorch.regularization import BaseRegularization
from neurotorch.regularization.connectome import DaleLawL2, ExecRatioTargetRegularization
from neurotorch.callbacks.lr_schedulers import LRSchedulerOnMetric
from neurotorch.visualisation.connectome import visualize_init_final_weights
from neurotorch.visualisation.time_series_visualisation import *


def train_with_params(
		true_time_series: np.ndarray or torch.Tensor,
		learning_rate: float = 1e-3,
		n_iterations: int = 100,
		forward_weights: Optional[torch.Tensor or np.ndarray] = None,
		std_weights: float = 1,
		dt: float = 1e-3,
		mu: Optional[float or torch.Tensor or np.ndarray] = 0.0,
		mean_mu: Optional[float] = 0.0,
		std_mu: Optional[float] = 1.0,
		r: Optional[float or torch.Tensor or np.ndarray] = 1.0,
		mean_r: Optional[float] = 1.0,
		std_r: Optional[float] = 1.0,
		tau: float = 1.0,
		learn_mu: bool = False,
		learn_r: bool = False,
		learn_tau: bool = False,
		device: torch.device = torch.device("cpu"),
		hh_init: str = "inputs",
		checkpoint_folder="./checkpoints",
		force_dale_law: bool = True
):
	if not torch.is_tensor(true_time_series):
		true_time_series = torch.tensor(true_time_series, dtype=torch.float32, device=device)
	x = true_time_series.T[np.newaxis, :]
	ws_layer = WilsonCowanLayer(
		x.shape[-1], x.shape[-1],
		forward_weights=forward_weights,
		std_weights=std_weights,
		forward_sign=0.5,
		dt=dt,
		r=r,
		mean_r=mean_r,
		std_r=std_r,
		mu=mu,
		mean_mu=mean_mu,
		std_mu=std_mu,
		tau=tau,
		learn_r=learn_r,
		learn_mu=learn_mu,
		learn_tau=learn_tau,
		hh_init=hh_init,
		device=device,
		name="WilsonCowan_layer1",
		force_dale_law=force_dale_law
	).build()

	# ws_layer_2 = deepcopy(ws_layer)  # only usefull if you're planning to use the second layer
	# ws_layer_2.name = "WilsonCowan_layer2"

	# The first model is for one layer while the second one is for two layers. Layers can be added as much as desired.
	model = nt.SequentialModel(layers=[ws_layer], device=device, foresight_time_steps=x.shape[1] - 1)
	# model = nt.SequentialModel(layers=[ws_layer, ws_layer_2], device=device, foresight_time_steps=x.shape[1] - 1)
	model.build()

	# Regularization on the connectome can be applied on one connectome or on all connectomes (or none).
	if force_dale_law:
		regularisation = ExecRatioTargetRegularization(ws_layer.get_sign_parameters(), exec_target_ratio=0.8)
		optimizer_reg = torch.optim.Adam(regularisation.parameters(), lr=5e-3)
	else:
		regularisation = DaleLawL2(ws_layer.get_weights_parameters(), alpha=1, inh_ratio=0.5, rho=0.99)
		optimizer_reg = torch.optim.SGD(regularisation.parameters(), lr=5e-4)

	optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, maximize=True, weight_decay=0.1)

	checkpoint_manager = nt.CheckpointManager(
		checkpoint_folder,
		metric="train_loss",
		minimise_metric=False,
		save_freq=-1,
		save_best_only=True,
		start_save_at=int(0.98 * n_iterations),
	)
	convergence_time_getter = ConvergenceTimeGetter(metric='train_loss', threshold=0.95, minimize_metric=False)
	callbacks = [
		LRSchedulerOnMetric(
			'train_loss',
			metric_schedule=np.linspace(0.97, 1.0, 100),
			min_lr=learning_rate / 10,
			retain_progress=True,
		),
		checkpoint_manager,
		convergence_time_getter,
		EarlyStoppingThreshold(metric='train_loss', threshold=0.999, minimize_metric=False),
	]

	with torch.no_grad():
		W0 = ws_layer.forward_weights.clone().detach().cpu().numpy()
		if force_dale_law:
			sign0 = ws_layer.forward_sign.clone().detach().cpu().numpy()
		else:
			sign0 = None
		mu0 = ws_layer.mu.clone()
		r0 = ws_layer.r.clone()
		tau0 = ws_layer.tau.clone()
		if ws_layer.force_dale_law:
			ratio_sign_0 = (np.mean(torch.sign(ws_layer.forward_sign).detach().cpu().numpy()) + 1) / 2
		else:
			ratio_sign_0 = (np.mean(torch.sign(ws_layer.forward_weights).detach().cpu().numpy()) + 1) / 2
		print(f"ratio exec init: {ratio_sign_0 :.3f}")

	dataset = WSDataset(true_time_series.T)
	trainer = nt.trainers.RegressionTrainer(
		model,
		callbacks=callbacks,
		optimizer=optimizer,
		regularization_optimizer=optimizer_reg,
		criterion=nt.losses.PVarianceLoss(),
		regularization=regularisation,
		metrics=[regularisation],
	)
	trainer.train(
		DataLoader(dataset, shuffle=False, num_workers=0, pin_memory=device.type == "cpu"),
		n_iterations=n_iterations,
		exec_metrics_on_train=True,
		# load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR,
		force_overwrite=True,
	)

	x_pred = torch.concat([
		torch.unsqueeze(x[:, 0].clone(), dim=1).to(model.device),
		model.get_prediction_trace(torch.unsqueeze(x[:, 0].clone(), dim=1))
	], dim=1)
	loss = PVarianceLoss()(x_pred, x)

	out = {
		"pVar": loss.detach().item(),
		"W": ws_layer.forward_weights.detach().cpu().numpy(),
		"sign0": sign0,
		"mu": ws_layer.mu.detach().numpy(),
		"r": ws_layer.r.detach().numpy(),
		"W0": W0,
		"ratio_0": ratio_sign_0,
		"mu0": mu0.numpy(),
		"r0": r0.numpy(),
		"tau0": tau0.numpy(),
		"tau": ws_layer.tau.detach().numpy(),
		"x_pred": torch.squeeze(x_pred).detach().numpy().T
	}
	if ws_layer.force_dale_law:
		out["ratio_end"] = (np.mean(torch.sign(ws_layer.forward_sign).detach().cpu().numpy()) + 1) / 2
		out["sign"] = ws_layer.forward_sign.clone().detach().cpu().numpy()
	else:
		out["ratio_end"] = (np.mean(torch.sign(ws_layer.forward_weights).detach().cpu().numpy()) + 1) / 2
		out["sign"] = None

	return out


if __name__ == '__main__':
	ts = np.load('timeSeries_2020_12_16_cr3_df.npy')
	n_neurons, n_shape = ts.shape
	sample_size = 200
	sample = np.random.randint(n_neurons, size=sample_size)
	data = ts[sample, :]

	sigma = 30

	for neuron in range(data.shape[0]):
		data[neuron, :] = gaussian_filter1d(data[neuron, :], sigma=sigma)
		data[neuron, :] = data[neuron, :] - np.min(data[neuron, :])
		data[neuron, :] = data[neuron, :] / np.max(data[neuron, :])

	forward_weights = nt.init.dale_(torch.zeros(200, 200), inh_ratio=0.5, rho=0.2)

	res = train_with_params(
		true_time_series=data,
		learning_rate=1e-2,
		n_iterations=500,
		forward_weights=forward_weights,
		std_weights=1,
		dt=0.02,
		mu=0.0,
		mean_mu=0,
		std_mu=1,
		r=0.1,
		mean_r=0.2,
		std_r=0,
		tau=0.1,
		learn_mu=True,
		learn_r=True,
		learn_tau=True,
		device=torch.device("cpu"),
		hh_init="inputs",
		force_dale_law=True
	)

	print(f"initiale ratio {res['ratio_0']:.3f}, finale ratio {res['ratio_end']:.3f}")
	fig, axes = plt.subplots(2, 2, figsize=(12, 8))
	visualize_init_final_weights(
		res["W0"], res["W"],
		fig=fig, axes=axes[0],
		show=False,
	)
	axes[0, 0].set_title("Initial weights, ratio exec {:.3f}".format(res["ratio_0"]))
	axes[0, 1].set_title("Final weights, ratio exec {:.3f}".format(res["ratio_end"]))
	sort_idx = np.argsort(res["sign"].ravel())
	axes[1, 0].plot(res["sign0"].ravel()[sort_idx])
	axes[1, 0].set_title("Initial signs")
	axes[1, 1].plot(res["sign"].ravel()[sort_idx])
	axes[1, 1].set_title("Final signs")
	plt.show()

	viz = VisualiseKMeans(
		res["x_pred"].T,
		shape=nt.Size([
			nt.Dimension(406, nt.DimensionProperty.TIME, "time [s]"),
			nt.Dimension(200, nt.DimensionProperty.NONE, "Neuron [-]"),
		])
	)
	viz.plot_timeseries_comparison(data.T, title=f"Prediction", show=True)

	fig, axes = plt.subplots(1, 2, figsize=(12, 8))
	VisualiseKMeans(
		data,
		nt.Size([
			nt.Dimension(200, nt.DimensionProperty.NONE, "Neuron [-]"),
			nt.Dimension(406, nt.DimensionProperty.TIME, "time [s]")])
	).heatmap(fig=fig, ax=axes[0], title="True time series")
	VisualiseKMeans(
		res["x_pred"],
		nt.Size([
			nt.Dimension(200, nt.DimensionProperty.NONE, "Neuron [-]"),
			nt.Dimension(406, nt.DimensionProperty.TIME, "time [s]")])
	).heatmap(fig=fig, ax=axes[1], title="Predicted time series")
	plt.show()

	Visualise(
		res["x_pred"],
		nt.Size([
			nt.Dimension(200, nt.DimensionProperty.NONE, "Neuron [-]"),
			nt.Dimension(406, nt.DimensionProperty.TIME, "time [s]")
		])
	).animate(time_interval=0.1, forward_weights=res["W"], dt=0.1, show=True)

	for i in range(sample_size):
		plt.plot(data[i, :], label="True")
		plt.plot(res["x_pred"][i, :], label="Pred")
		plt.ylim([0, 1])
		plt.legend()
		plt.show()
