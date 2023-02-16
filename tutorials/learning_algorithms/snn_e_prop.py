import pprint

import neurotorch as nt
from neurotorch.callbacks.lr_schedulers import LRSchedulerOnMetric
from neurotorch.visualisation.connectome import visualize_init_final_weights
from neurotorch.visualisation.time_series_visualisation import *
from tutorials.learning_algorithms.dataset import get_dataloader


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
	lif_layer = nt.SpyLIFLayerLPF(
		x.shape[-1], params["n_aux_units"],
		dt=params["dt"],
		hh_init=params["hh_init"],
		force_dale_law=params["force_dale_law"],
		use_recurrent_connection=params["use_recurrent_connection"],
		device=device,
	).build()
	layers = [lif_layer, ]
	if params["add_out_layer"]:
		out_layer = nt.Linear(
			params["n_aux_units"], x.shape[-1],
			device=device,
			use_bias=False,
			activation=params["activation"],
		)
		layers.append(out_layer)
	checkpoint_manager = nt.CheckpointManager(
		checkpoint_folder="./data/tr_data/checkpoints_snn_e_prop",
		metric="val_p_var",
		minimise_metric=False,
		save_freq=max(1, int(n_iterations / 10)),
		start_save_at=max(1, int(n_iterations / 3)),
		save_best_only=True,
	)
	model = nt.SequentialRNN(
		layers=layers,
		device=device,
		foresight_time_steps=dataset.n_time_steps - 1,
		out_memory_size=dataset.n_time_steps - 1,
		hh_memory_size=1,
		checkpoint_folder=checkpoint_manager.checkpoint_folder,
	).build()
	la = nt.Eprop(
		alpha=1e-3,
		gamma=1e-3,
		params_lr=1e-5,
		output_params_lr=2e-5,
		default_optimizer_cls=torch.optim.AdamW,
		default_optim_kwargs={"weight_decay": 1e-3, "lr": 1e-6},
		eligibility_traces_norm_clip_value=1.0,
		grad_norm_clip_value=1.0,
		learning_signal_norm_clip_value=1.0,
		feedback_weights_norm_clip_value=1.0,
		feedbacks_gen_strategy="randn",
	)
	lr_scheduler = LRSchedulerOnMetric(
		'val_p_var',
		metric_schedule=np.linspace(kwargs.get("lr_schedule_start", 0.5), 1.0, 100),
		min_lr=[1e-7, 2e-7],
		retain_progress=True,
		priority=la.priority + 1,
	)
	callbacks = [la, checkpoint_manager, lr_scheduler]
	
	with torch.no_grad():
		W0 = nt.to_numpy(lif_layer.forward_weights.clone())
		if params["force_dale_law"]:
			sign0 = nt.to_numpy(lif_layer.forward_sign.clone())
		else:
			sign0 = None
		if lif_layer.force_dale_law:
			ratio_sign_0 = (np.mean(nt.to_numpy(torch.sign(lif_layer.forward_sign))) + 1) / 2
		else:
			ratio_sign_0 = (np.mean(nt.to_numpy(torch.sign(lif_layer.forward_weights))) + 1) / 2
	
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
	history.plot(save_path=f"data/figures/snn_eprop/tr_history.png", show=True)
	
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
		"W"                   : nt.to_numpy(lif_layer.forward_weights),
		"sign0"               : sign0,
		"W0"                  : W0,
		"ratio_0"             : ratio_sign_0,
		"x_pred"              : nt.to_numpy(torch.squeeze(x_pred)),
		"original_time_series": dataset.full_time_series.squeeze(),
		"force_dale_law"      : params["force_dale_law"],
	}
	if lif_layer.force_dale_law:
		out["ratio_end"] = (nt.to_numpy(np.mean(torch.sign(lif_layer.forward_sign))) + 1) / 2
		out["sign"] = nt.to_numpy(lif_layer.forward_sign.clone())
	else:
		out["ratio_end"] = (np.mean(nt.to_numpy(torch.sign(lif_layer.forward_weights))) + 1) / 2
		out["sign"] = None
	
	return out


if __name__ == '__main__':
	res = train_with_params(
		params={
			# "filename": "ts_nobaselines_fish3.npy",
			# "filename": "corrected_data.npy",
			# "filename": "curbd_Adata.npy",
			"filename"                 : None,
			"smoothing_sigma"          : 5.0,
			"n_units"                  : 500,
			"n_aux_units"              : 500,
			"n_time_steps"             : -1,
			"dataset_length"           : 1,
			"dataset_randomize_indexes": False,
			"force_dale_law"           : False,
			"learn_mu"                 : True,
			"learn_r"                  : True,
			"learn_tau"                : True,
			"use_recurrent_connection" : False,
		},
		n_iterations=2000,
		device=torch.device("cpu"),
		force_overwrite=False,
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
		res["x_pred"],
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
		filename=f"data/figures/snn_eprop/timeseries_comparison_report.png",
		show=True,
		dpi=600,
	)
	
	fig, axes = plt.subplots(1, 2, figsize=(12, 8))
	viz_kmeans = VisualiseKMeans(
		res["original_time_series"],
		nt.Size(
			[
				nt.Dimension(None, nt.DimensionProperty.TIME, "Time [s]"),
				nt.Dimension(None, nt.DimensionProperty.NONE, "Neuron [-]"),
			]
		)
	)
	viz_kmeans.heatmap(fig=fig, ax=axes[0], title="True time series")
	Visualise(
		viz_kmeans.permute_timeseries(res["x_pred"]),
		nt.Size(
			[
				nt.Dimension(None, nt.DimensionProperty.TIME, "Time [s]"),
				nt.Dimension(None, nt.DimensionProperty.NONE, "Neuron [-]"),
			]
		)
	).heatmap(fig=fig, ax=axes[1], title="Predicted time series")
	plt.savefig("data/figures/snn_eprop/heatmap.png")
	plt.show()
	Visualise(
		res["x_pred"],
		nt.Size(
			[
				nt.Dimension(None, nt.DimensionProperty.TIME, "Time [s]"),
				nt.Dimension(None, nt.DimensionProperty.NONE, "Neuron [-]"),
			]
		)
	).animate(
		time_interval=0.1,
		forward_weights=res["W"],
		dt=0.1,
		show=False,
		filename="data/figures/snn_eprop/animation.mp4"
	)
