import pprint

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import neurotorch as nt
from neurotorch.callbacks.convergence import ConvergenceTimeGetter
from neurotorch.callbacks.early_stopping import EarlyStoppingThreshold
from neurotorch.callbacks.events import EventOnMetricThreshold
from neurotorch.callbacks.lr_schedulers import LRSchedulerOnMetric
from neurotorch.learning_algorithms.curbd import CURBD
from neurotorch.modules.layers import WilsonCowanLayer
from neurotorch.regularization.connectome import DaleLawL2, ExecRatioTargetRegularization
from neurotorch.utils import hash_params
from neurotorch.visualisation.connectome import visualize_init_final_weights
from neurotorch.visualisation.time_series_visualisation import *
from tutorials.learning_algorithms.dataset import get_dataloader
from tutorials.time_series_forecasting_wilson_cowan.dataset import WSDataset


def increase_trainer_iteration_event(trainer, **kwargs):
	trainer.update_state_(n_iterations=int(trainer.state.n_iterations * kwargs.get("delta_iterations", 1.5)))


def increase_n_time_steps_event(trainer, **kwargs):
	dataset = trainer.state.objects["train_dataloader"].dataset
	old_n_time_steps = dataset.n_time_steps
	dataset.n_time_steps += kwargs.get("delta_time_steps", 10)
	trainer.model.out_memory_size = dataset.n_time_steps - 1
	trainer.model.foresight_time_steps = dataset.n_time_steps - 1
	if old_n_time_steps != dataset.n_time_steps:
		trainer.update_state_(
			n_iterations=int(
				trainer.state.n_iterations * kwargs.get("delta_iterations", 1.5) + trainer.state.iteration
			)
		)
	return f"({dataset.n_time_steps=})"
	

def set_default_param(**kwargs):
	kwargs.setdefault("filename", None)
	kwargs.setdefault("dataset_length", -1)
	kwargs.setdefault("n_time_steps", -1)
	kwargs.setdefault("target_skip_first", True)
	kwargs.setdefault("dataset_randomize_indexes", False)
	kwargs.setdefault("rm_dead_units", True)
	kwargs.setdefault("smoothing_sigma", 20.0)
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
	kwargs.setdefault("hh_init", "inputs")
	kwargs.setdefault("force_dale_law", True)
	kwargs.setdefault("seed", 0)
	kwargs.setdefault("n_units", 200)
	kwargs.setdefault("n_layers", 1)
	kwargs.setdefault("learning_algorithm", kwargs.pop("la", "bptt"))
	kwargs.setdefault("forward_sign", 0.5)
	kwargs.setdefault("activation", "tanh")
	return kwargs


def make_learning_algorithm(**kwargs):
	la_name = kwargs.get("learning_algorithm", kwargs.get("la", "bptt")).lower()
	if la_name == "bptt":
		optimizer = torch.optim.AdamW(
			kwargs["model"].parameters(), lr=kwargs["learning_rate"], maximize=True,
			weight_decay=kwargs.get("weight_decay", 0.1)
		)
		learning_algorithm = nt.BPTT(optimizer=optimizer, criterion=nt.losses.PVarianceLoss())
	elif la_name == "eprop":
		learning_algorithm = nt.Eprop(criterion=nt.losses.PVarianceLoss())
	elif la_name == "tbptt":
		optimizer = torch.optim.AdamW(
			kwargs["model"].parameters(), lr=kwargs["learning_rate"], maximize=True,
			weight_decay=kwargs.get("weight_decay", 0.1)
		)
		learning_algorithm = nt.TBPTT(
			optimizer=optimizer, criterion=nt.losses.PVarianceLoss(),
			auto_backward_time_steps_ratio=kwargs.get("auto_backward_time_steps_ratio", 0.1)
		)
	elif la_name == "weakrls":
		learning_algorithm = nt.WeakRLS(
			criterion=nt.losses.PVarianceLoss(),
			# criterion=torch.nn.MSELoss(),
			# device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
			device=torch.device("cpu"),
			reduction='mean',
			is_recurrent=True,
		)
	elif la_name == "rls":
		learning_algorithm = nt.RLS(
			params=[kwargs["model"].get_layer().forward_weights],
			criterion=nt.losses.PVarianceLoss(),
			# device=torch.device("cpu"),
			strategy=kwargs.get("rls_strategy", "inputs"),
			is_recurrent=True,
			auto_backward_time_steps_ratio=kwargs.get("auto_backward_time_steps_ratio", 0)
		)
	elif la_name == "curbd":
		learning_algorithm = CURBD(
			params=[kwargs["model"].get_layer().forward_weights],
			criterion=nt.losses.PVarianceLoss(),
			device=torch.device("cpu"),
		)
	else:
		raise ValueError(f"Unknown learning algorithm: {la_name}")
	return learning_algorithm


def train_with_params(
		params: Optional[dict] = None,
		n_iterations: int = 100,
		device: torch.device = torch.device("cpu"),
		checkpoint_folder="./data/tr_data/checkpoints",
		**kwargs
):
	if params is None:
		params = {}
	params = set_default_param(**params)
	kwargs.setdefault("force_overwrite", False)
	torch.manual_seed(params["seed"])
	checkpoints_name = str(hash_params(params))
	checkpoint_folder = f"{checkpoint_folder}/{checkpoints_name}"
	os.makedirs(checkpoint_folder, exist_ok=True)
	dataloader = get_dataloader(
		batch_size=kwargs.get("batch_size", 512), verbose=True, n_workers=kwargs.get("n_workers"), **params
	)
	dataset = dataloader.dataset
	x = dataset.full_time_series
	# forward_weights = nt.init.dale_(torch.zeros(params["n_units"], params["n_units"]), inh_ratio=0.5, rho=0.2)
	forward_weights = 1.5 * torch.randn(x.shape[-1], x.shape[-1]) / np.sqrt(x.shape[-1])
	ws_layer = nt.WilsonCowanLayer(
		x.shape[-1], x.shape[-1],
		forward_weights=forward_weights,
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
	).build()
	layers = [ws_layer]
	for i in range(1, params["n_layers"]):
		ws_layer_i = deepcopy(ws_layer)
		ws_layer_i.name = f"WilsonCowan_layer{i+1}"
		layers.append(ws_layer_i)

	model = nt.SequentialModel(layers=layers, device=device, foresight_time_steps=dataset.n_time_steps - 1).build()

	# Regularization on the connectome can be applied on one connectome or on all connectomes (or none).
	if params["force_dale_law"]:
		regularisation = ExecRatioTargetRegularization(ws_layer.get_sign_parameters(), exec_target_ratio=0.8)
		optimizer_reg = torch.optim.Adam(regularisation.parameters(), lr=5e-3)
	else:
		regularisation = DaleLawL2(ws_layer.get_weights_parameters(), alpha=0.3, inh_ratio=0.5, rho=0.99)
		optimizer_reg = torch.optim.SGD(regularisation.parameters(), lr=5e-4)
	
	checkpoint_manager = nt.CheckpointManager(
		checkpoint_folder,
		metric="train_loss",
		minimise_metric=False,
		save_freq=-1,
		save_best_only=True,
		start_save_at=int(0.98 * n_iterations),
	)
	convergence_time_getter = ConvergenceTimeGetter(metric='train_loss', threshold=0.95, minimize_metric=False)
	learning_algorithm = make_learning_algorithm(**params, model=model)
	learning_algorithms = [learning_algorithm, ]
	params_not_taken = list(set(model.parameters()) - set(learning_algorithm.params))
	if len(params_not_taken) > 0 and params.get("add_aux_tbptt", False):
		aux_optimizer = torch.optim.Adam(
			params_not_taken, lr=params["learning_rate"], maximize=True,
			weight_decay=params.get("weight_decay", 0.1)
		)
		aux_learning_algorithm = nt.TBPTT(
			optimizer=aux_optimizer, criterion=nt.losses.PVarianceLoss(),
			backward_time_steps=1, optim_time_steps=1, priority=learning_algorithm.priority - 1
		)
		learning_algorithms.append(aux_learning_algorithm)
	callbacks = [
		# LRSchedulerOnMetric(
		# 	'train_loss',
		# 	metric_schedule=np.linspace(0.92, 1.0, 100),
		# 	min_lr=params["learning_rate"] / 10,
		# 	retain_progress=True,
		# ),
		*learning_algorithms,
		checkpoint_manager,
		convergence_time_getter,
		EarlyStoppingThreshold(metric='train_loss', threshold=0.99, minimize_metric=False),
		# EventOnMetricThreshold(
		# 	metric_name='train_loss', threshold=0.8, minimize_metric=False,
		# 	event=increase_trainer_iteration_event, do_once=False, event_kwargs={"delta_iterations": 2}
		# ),
		# EventOnMetricThreshold(
		# 	metric_name='train_loss', threshold=0.8, minimize_metric=False,
		# 	event=increase_n_time_steps_event, do_once=False,
		# 	event_kwargs={"delta_time_steps": 100, "delta_iterations": 1.1},
		# 	name="increase_n_time_steps_event",
		# ),
	]

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
		# regularization_optimizer=optimizer_reg,
		# regularization=regularisation,
		# metrics=[regularisation],
	)
	print(f"{trainer}")
	history = trainer.train(
		dataloader,
		n_iterations=n_iterations,
		exec_metrics_on_train=True,
		load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR,
		force_overwrite=kwargs["force_overwrite"],
	)
	history.plot(show=True)
	
	model.foresight_time_steps = x.shape[1] - 1
	model.out_memory_size = model.foresight_time_steps
	x_pred = torch.concat([
		torch.unsqueeze(x[:, 0].clone(), dim=1).to(model.device),
		model.get_prediction_trace(torch.unsqueeze(x[:, 0].clone(), dim=1))
	], dim=1)
	loss = PVarianceLoss()(x_pred.to(x.device), x)

	out = {
		"params": params,
		"pVar": nt.to_numpy(loss.item()),
		"W": nt.to_numpy(ws_layer.forward_weights),
		"sign0": sign0,
		"mu": nt.to_numpy(ws_layer.mu),
		"r": nt.to_numpy(ws_layer.r),
		"W0": W0,
		"ratio_0": ratio_sign_0,
		"mu0": nt.to_numpy(mu0),
		"r0": nt.to_numpy(r0),
		"tau0": nt.to_numpy(tau0),
		"tau": nt.to_numpy(ws_layer.tau),
		"x_pred": nt.to_numpy(torch.squeeze(x_pred).T),
		"original_time_series": dataset.original_series,
		"force_dale_law": params["force_dale_law"],
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
			"filename": "ts_nobaselines_fish3.npy",
			# "filename": "corrected_data.npy",
			# "filename": "curbd_Adata.npy",
			"smoothing_sigma": 10.0,
			"n_units": 200,
			"n_time_steps": -1,
			"dataset_length": 1,
			"dataset_randomize_indexes": False,
			"force_dale_law": False,
			"learning_algorithm": "RLS",
			"auto_backward_time_steps_ratio": 0.0,
			"weight_decay": 1e-5,
			"learn_mu": True,
			"learn_r": True,
			"learn_tau": True,
			"activation": "sigmoid",
			"rls_strategy": "inputs",
			"add_aux_tbptt": True,
		},
		n_iterations=100,
		device=torch.device("cuda"),
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
		res["x_pred"].T,
		shape=nt.Size([
			nt.Dimension(None, nt.DimensionProperty.TIME, "Time [s]"),
			nt.Dimension(None, nt.DimensionProperty.NONE, "Neuron [-]"),
		])
	)
	viz.plot_timeseries_comparison(
		res["original_time_series"].T, title=f"Prediction", show=True, filename="figures/prediction.png"
	)

	fig, axes = plt.subplots(1, 2, figsize=(12, 8))
	VisualiseKMeans(
		res["original_time_series"],
		nt.Size([
			nt.Dimension(None, nt.DimensionProperty.NONE, "Neuron [-]"),
			nt.Dimension(None, nt.DimensionProperty.TIME, "time [s]")])
	).heatmap(fig=fig, ax=axes[0], title="True time series")
	VisualiseKMeans(
		res["x_pred"],
		nt.Size([
			nt.Dimension(None, nt.DimensionProperty.NONE, "Neuron [-]"),
			nt.Dimension(None, nt.DimensionProperty.TIME, "time [s]")])
	).heatmap(fig=fig, ax=axes[1], title="Predicted time series")
	plt.savefig("figures/heatmap.png")
	plt.show()

	Visualise(
		res["x_pred"],
		nt.Size([
			nt.Dimension(None, nt.DimensionProperty.NONE, "Neuron [-]"),
			nt.Dimension(None, nt.DimensionProperty.TIME, "time [s]")
		])
	).animate(time_interval=0.1, forward_weights=res["W"], dt=0.1, show=False, filename="figures/animation.mp4")

