import pprint

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import neurotorch as nt
from neurotorch.callbacks.convergence import ConvergenceTimeGetter
from neurotorch.callbacks.early_stopping import EarlyStoppingThreshold
from neurotorch.callbacks.events import EventOnMetricThreshold
from neurotorch.callbacks.lr_schedulers import LRSchedulerOnMetric
from neurotorch.modules.layers import WilsonCowanLayer
from neurotorch.regularization.connectome import DaleLawL2, ExecRatioTargetRegularization
from neurotorch.utils import hash_params
from neurotorch.visualisation.connectome import visualize_init_final_weights
from neurotorch.visualisation.time_series_visualisation import *
from tutorials.learning_algorithms.dataset import get_dataloader
from tutorials.time_series_forecasting_wilson_cowan.dataset import WSDataset


def set_default_param(**kwargs):
	kwargs.setdefault("filename", None)
	kwargs.setdefault("dataset_length", -1)
	kwargs.setdefault("n_time_steps", -1)
	kwargs.setdefault("target_skip_first", True)
	kwargs.setdefault("dataset_randomize_indexes", False)
	kwargs.setdefault("rm_dead_units", True)
	kwargs.setdefault("smoothing_sigma", 15.0)
	kwargs.setdefault("learning_rate", 1e-2)
	kwargs.setdefault("std_weights", 1)
	kwargs.setdefault("dt", 0.02)
	kwargs.setdefault("mu", 0.0)
	kwargs.setdefault("mean_mu", 0.0)
	kwargs.setdefault("std_mu", 1.0)
	kwargs.setdefault("r", 0.1)
	kwargs.setdefault("mean_r", 0.5)
	kwargs.setdefault("std_r", 0.4)
	kwargs.setdefault("tau", 0.1)
	kwargs.setdefault("learn_mu", True)
	kwargs.setdefault("learn_r", True)
	kwargs.setdefault("learn_tau", True)
	kwargs.setdefault("force_dale_law", False)
	kwargs.setdefault("seed", 0)
	kwargs.setdefault("n_units", 200)
	kwargs.setdefault("n_aux_units", kwargs["n_units"])
	kwargs.setdefault("use_recurrent_connection", False)
	kwargs.setdefault("add_out_layer", True)
	if not kwargs["add_out_layer"]:
		kwargs["n_aux_units"] = kwargs["n_units"]
	kwargs.setdefault("forward_sign", 0.5)
	kwargs.setdefault("activation", "sigmoid")
	kwargs.setdefault("hh_init", "inputs" if kwargs["n_aux_units"] == kwargs["n_units"] else "random")
	return kwargs


def train_with_params(
		params: Optional[dict] = None,
		n_iterations: int = 100,
		device: torch.device = torch.device("cpu"),
		**kwargs
):
	if params is None:
		params = {}
	params = set_default_param(**params)
	kwargs.setdefault("force_overwrite", False)
	torch.manual_seed(params["seed"])
	dataloader = get_dataloader(
		batch_size=kwargs.get("batch_size", 512), verbose=True, n_workers=kwargs.get("n_workers"), **params
	)
	dataset = dataloader.dataset
	x = dataset.full_time_series
	ws_layer = nt.WilsonCowanLayer(
		x.shape[-1], params["n_aux_units"],
		std_weights=params["std_weights"],
		forward_sign=params["forward_sign"],
		dt=params["dt"],
		r=params["r"],
		mean_r=params["mean_r"],
		std_r=params["std_r"],
		mu=params["mu"],
		mean_mu=params["mean_mu"],
		std_mu=params["std_mu"],
		tau=params["tau"],
		learn_r=params["learn_r"],
		learn_mu=params["learn_mu"],
		learn_tau=params["learn_tau"],
		hh_init=params["hh_init"],
		device=device,
		name="WilsonCowan_layer1",
		force_dale_law=params["force_dale_law"],
		activation=params["activation"],
		use_recurrent_connection=params["use_recurrent_connection"],
	).build()
	layers = [ws_layer]
	if params["add_out_layer"]:
		out_layer = nt.Linear(
			params["n_aux_units"], x.shape[-1],
			device=device,
			use_bias=False,
			activation=params["activation"],
		)
		layers.append(out_layer)
	checkpoint_manager = nt.CheckpointManager(
		checkpoint_folder="./checkpoints_wc_e_prop",
		# metric="val_loss",
		metric="val_p_var",
		minimise_metric=False,
		save_freq=max(1, int(n_iterations / 10)),
		save_best_only=True,
	)
	model = nt.SequentialRNN(
		layers=layers,
		device=device,
		foresight_time_steps=dataset.n_time_steps - 1,
		out_memory_size=dataset.n_time_steps - 1,
		checkpoint_folder=checkpoint_manager.checkpoint_folder,
	).build()
	la = nt.Eprop(alpha=0.9, default_optim_kwargs={"weight_decay": 1e-5, "lr": 1e-3})
	# la.DEFAULT_OPTIMIZER_CLS = torch.optim.SGD
	callbacks = [la, checkpoint_manager]
	
	with torch.no_grad():
		W0 = ws_layer.forward_weights.clone().detach().cpu().numpy()
		if params["force_dale_law"]:
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
	
	trainer = nt.trainers.Trainer(
		model,
		predict_method="get_prediction_trace",
		callbacks=callbacks,
		metrics=[nt.metrics.RegressionMetrics(model, "p_var")],
	)
	print(f"{trainer}")
	history = trainer.train(
		dataloader,
		dataloader,
		n_iterations=n_iterations,
		exec_metrics_on_train=False,
		load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR,
		force_overwrite=kwargs["force_overwrite"],
	)
	history.plot(show=True)

	model.eval()
	model.load_checkpoint(checkpoint_manager.checkpoints_meta_path)
	model.foresight_time_steps = x.shape[1] - 1
	model.out_memory_size = model.foresight_time_steps
	x_pred = torch.concat(
		[
			torch.unsqueeze(x[:, 0].clone(), dim=1).to(model.device),
			model.get_prediction_trace(torch.unsqueeze(x[:, 0].clone(), dim=1))
		], dim=1
	)
	loss = PVarianceLoss()(x_pred.to(x.device), x)
	
	out = {
		"params"              : params,
		"pVar"                : nt.to_numpy(loss.item()),
		"W"                   : nt.to_numpy(ws_layer.forward_weights),
		"sign0"               : sign0,
		"mu"                  : nt.to_numpy(ws_layer.mu),
		"r"                   : nt.to_numpy(ws_layer.r),
		"W0"                  : W0,
		"ratio_0"             : ratio_sign_0,
		"mu0"                 : nt.to_numpy(mu0),
		"r0"                  : nt.to_numpy(r0),
		"tau0"                : nt.to_numpy(tau0),
		"tau"                 : nt.to_numpy(ws_layer.tau),
		"x_pred"              : nt.to_numpy(torch.squeeze(x_pred).T),
		"original_time_series": dataset.full_time_series,
		"force_dale_law"      : params["force_dale_law"],
	}
	if ws_layer.force_dale_law:
		out["ratio_end"] = (np.mean(torch.sign(ws_layer.forward_sign).detach().cpu().numpy()) + 1) / 2
		out["sign"] = ws_layer.forward_sign.clone().detach().cpu().numpy()
	else:
		out["ratio_end"] = (np.mean(torch.sign(ws_layer.forward_weights).detach().cpu().numpy()) + 1) / 2
		out["sign"] = None
	
	return out


if __name__ == '__main__':
	res = train_with_params(
		params={
			# "filename": "ts_nobaselines_fish3.npy",
			# "filename": "corrected_data.npy",
			# "filename": "curbd_Adata.npy",
			"filename"                      : None,
			"smoothing_sigma"               : 15.0,
			"n_units"                       : 200,
			"n_aux_units"                   : 200,
			"n_time_steps"                  : -1,
			"dataset_length"                : 1,
			"dataset_randomize_indexes"     : False,
			"force_dale_law"                : False,
			"learn_mu"                      : True,
			"learn_r"                       : True,
			"learn_tau"                     : True,
			"use_recurrent_connection"      : False,
		},
		n_iterations=300,
		device=torch.device("cpu"),
		force_overwrite=True,
		batch_size=1,
	)
	pprint.pprint({k: v for k, v in res.items() if isinstance(v, (int, float, str, bool))})
	
	if res["force_dale_law"]:
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
	
	viz = Visualise(
		res["x_pred"].T,
		shape=nt.Size(
			[
				nt.Dimension(None, nt.DimensionProperty.TIME, "Time [s]"),
				nt.Dimension(None, nt.DimensionProperty.NONE, "Neuron [-]"),
			]
		)
	)
	viz.plot_timeseries_comparison_report(
		res["original_time_series"],
		title=f"Prediction",
		show=True,
		dpi=600,
	)
	viz.plot_timeseries_comparison(
		res["original_time_series"], title=f"Prediction", show=True, filename="figures/prediction.png"
	)
	
	fig, axes = plt.subplots(1, 2, figsize=(12, 8))
	VisualiseKMeans(
		res["original_time_series"].T,
		nt.Size(
			[
				nt.Dimension(None, nt.DimensionProperty.NONE, "Neuron [-]"),
				nt.Dimension(None, nt.DimensionProperty.TIME, "time [s]")]
		)
	).heatmap(fig=fig, ax=axes[0], title="True time series")
	VisualiseKMeans(
		res["x_pred"],
		nt.Size(
			[
				nt.Dimension(None, nt.DimensionProperty.NONE, "Neuron [-]"),
				nt.Dimension(None, nt.DimensionProperty.TIME, "time [s]")]
		)
	).heatmap(fig=fig, ax=axes[1], title="Predicted time series")
	plt.savefig("figures/heatmap.png")
	plt.show()
	
	Visualise(
		res["x_pred"],
		nt.Size(
			[
				nt.Dimension(None, nt.DimensionProperty.NONE, "Neuron [-]"),
				nt.Dimension(None, nt.DimensionProperty.TIME, "time [s]")
			]
		)
	).animate(time_interval=0.1, forward_weights=res["W"], dt=0.1, show=False, filename="figures/animation.mp4")

