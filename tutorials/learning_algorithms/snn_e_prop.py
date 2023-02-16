import pprint

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

import neurotorch as nt
from neurotorch.callbacks.convergence import ConvergenceTimeGetter
from neurotorch.callbacks.early_stopping import EarlyStoppingThreshold
from neurotorch.callbacks.events import EventOnMetricThreshold
from neurotorch.callbacks.lr_schedulers import LRSchedulerOnMetric
from neurotorch.dimension import SizeTypes
from neurotorch.modules import HeavisideSigmoidApprox
from neurotorch.modules.layers import WilsonCowanLayer, BaseNeuronsLayer
from neurotorch.regularization.connectome import DaleLawL2, ExecRatioTargetRegularization
from neurotorch.utils import hash_params, format_pseudo_rn_seed
from neurotorch.visualisation.connectome import visualize_init_final_weights
from neurotorch.visualisation.time_series_visualisation import *
from tutorials.learning_algorithms.dataset import get_dataloader
from tutorials.time_series_forecasting_wilson_cowan.dataset import WSDataset


class SpyLIFLayerLowPassFilter(BaseNeuronsLayer):
	"""
	The SpyLIF dynamics is a more complex variant of the LIF dynamics (class :class:`LIFLayer`) allowing it to have a
	greater power of expression. This variant is also inspired by Neftci :cite:t:`neftci_surrogate_2019` and also
	contains  two differential equations like the SpyLI dynamics :class:`SpyLI`. The equation :eq:`SpyLIF_I` presents
	the synaptic current update equation with euler integration while the equation :eq:`SpyLIF_V` presents the
	synaptic potential update.

	.. math::
		:label: SpyLIF_I

		\\begin{equation}
			I_{\\text{syn}, j}^{t+\\Delta t} = \\alpha I_{\text{syn}, j}^{t} + \\sum_{i}^{N} W_{ij}^{\\text{rec}} z_i^t
			+ \\sum_i^{N} W_{ij}^{\\text{in}} x_i^{t+\\Delta t}
		\\end{equation}


	.. math::
		:label: SpyLIF_V

		\\begin{equation}
			V_j^{t+\\Delta t} = \\left(\\beta V_j^t + I_{\\text{syn}, j}^{t+\\Delta t}\\right) \\left(1 - z_j^t\\right)
		\\end{equation}


	.. math::
		:label: spylif_alpha

		\\begin{equation}
			\\alpha = e^{-\\frac{\\Delta t}{\\tau_{\\text{syn}}}}
		\\end{equation}

	with :math:`\\tau_{\\text{syn}}` being the decay time constant of the synaptic current.

	.. math::
		:label: spylif_beta

		\\begin{equation}
			\\beta = e^{-\\frac{\\Delta t}{\\tau_{\\text{mem}}}}
		\\end{equation}

	with :math:`\\tau_{\\text{syn}}` being the decay time constant of the synaptic current.

	The output of neuron :math:`j` at time :math:`t` denoted :math:`z_j^t` is defined by the equation :eq:`spylif_z` .

	.. math::
		:label: spylif_z

		z_j^t = H(V_j^t - V_{\\text{th}})

	where :math:`V_{\\text{th}}` denotes the activation threshold of the neuron and the function :math:`H(\\cdot)`
	is the Heaviside function defined as :math:`H(x) = 1` if :math:`x \\geq 0` and :math:`H(x) = 0` otherwise.

	SpyTorch library: https://github.com/surrogate-gradient-learning/spytorch.

	The variables of the equations :eq:`SpyLIF_I` and :eq:`SpyLIF_V` are described by the following definitions:

		- :math:`N` is the number of neurons in the layer.
		- :math:`I_{\\text{syn}, j}^{t}` is the synaptic current of neuron :math:`j` at time :math:`t`.
		- :math:`V_j^t` is the synaptic potential of the neuron :math:`j` at time :math:`t`.
		- :math:`\\Delta t` is the integration time step.
		- :math:`z_j^t` is the spike of the neuron :math:`j` at time :math:`t`.
		- :math:`\\alpha` is the decay constant of the synaptic current over time (equation :eq:`spylif_alpha`).
		- :math:`\\beta` is the decay constant of the membrane potential over time (equation :eq:`spylif_beta`).
		- :math:`W_{ij}^{\\text{rec}}` is the recurrent weight of the neuron :math:`i` to the neuron :math:`j`.
		- :math:`W_{ij}^{\\text{in}}` is the input weight of the neuron :math:`i` to the neuron :math:`j`.
		- :math:`x_i^{t}` is the input of the neuron :math:`i` at time :math:`t`.

	:Attributes:
		- :attr:`alpha` (torch.nn.Parameter): Decay constant of the synaptic current over time (equation :eq:`spylif_alpha`).
		- :attr:`beta` (torch.nn.Parameter): Decay constant of the membrane potential over time (equation :eq:`spylif_beta`).
		- :attr:`threshold` (torch.nn.Parameter): Activation threshold of the neuron (:math:`V_{\\text{th}}`).
		- :attr:`gamma` (torch.nn.Parameter): Slope of the Heaviside function (:math:`\\gamma`).
	"""
	
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			use_recurrent_connection: bool = True,
			use_rec_eye_mask: bool = False,
			dt: float = 1e-3,
			device: Optional[torch.device] = None,
			**kwargs
	):
		"""
		Constructor for the SpyLIF layer.

		:param input_size: The size of the input.
		:type input_size: Optional[SizeTypes]
		:param output_size: The size of the output.
		:type output_size: Optional[SizeTypes]
		:param name: The name of the layer.
		:type name: Optional[str]
		:param use_recurrent_connection: Whether to use the recurrent connection.
		:type use_recurrent_connection: bool
		:param use_rec_eye_mask: Whether to use the recurrent eye mask.
		:type use_rec_eye_mask: bool
		:param spike_func: The spike function to use.
		:type spike_func: Callable[[torch.Tensor], torch.Tensor]
		:param learning_type: The learning type to use.
		:type learning_type: LearningType
		:param dt: Time step (Euler's discretisation).
		:type dt: float
		:param device: The device to use.
		:type device: Optional[torch.device]
		:param kwargs: The keyword arguments for the layer.

		:keyword float tau_syn: The synaptic time constant :math:`\\tau_{\\text{syn}}`. Default: 5.0 * dt.
		:keyword float tau_mem: The membrane time constant :math:`\\tau_{\\text{mem}}`. Default: 10.0 * dt.
		:keyword float threshold: The threshold potential :math:`V_{\\text{th}}`. Default: 1.0.
		:keyword float gamma: The multiplier of the derivative of the spike function :math:`\\gamma`. Default: 100.0.
		:keyword float spikes_regularization_factor: The regularization factor for the spikes. Higher this factor is,
			the more the network will tend to spike less. Default: 0.0.

		"""
		self.spike_func = HeavisideSigmoidApprox
		super(SpyLIFLayerLowPassFilter, self).__init__(
			input_size=input_size,
			output_size=output_size,
			name=name,
			use_recurrent_connection=use_recurrent_connection,
			use_rec_eye_mask=use_rec_eye_mask,
			dt=dt,
			device=device,
			**kwargs
		)
		
		self.alpha = nn.Parameter(
			torch.tensor(np.exp(-dt / self.kwargs["tau_syn"]), dtype=torch.float32, device=self.device),
			requires_grad=False
		)
		self.beta = nn.Parameter(
			torch.tensor(np.exp(-dt / self.kwargs["tau_mem"]), dtype=torch.float32, device=self.device),
			requires_grad=False
		)
		self.threshold = nn.Parameter(
			torch.tensor(self.kwargs["threshold"], dtype=torch.float32, device=self.device),
			requires_grad=False
		)
		self.gamma = nn.Parameter(
			torch.tensor(self.kwargs["gamma"], dtype=torch.float32, device=self.device),
			requires_grad=False
		)
		self._regularization_l1 = torch.tensor(0.0, dtype=torch.float32, device=self.device)
		self._n_spike_per_neuron = torch.zeros(int(self.output_size), dtype=torch.float32, device=self.device)
		self._total_count = 0
		self._use_low_pass_filter = self.kwargs["use_low_pass_filter"]
		self._low_pass_filter_alpha = self.kwargs["low_pass_filter_alpha"]
	
	def _set_default_kwargs(self):
		self.kwargs.setdefault("tau_syn", 5.0 * self.dt)
		self.kwargs.setdefault("tau_mem", 10.0 * self.dt)
		self.kwargs.setdefault("threshold", 1.0)
		self.kwargs.setdefault("gamma", 100.0)
		self.kwargs.setdefault("spikes_regularization_factor", 0.0)
		self.kwargs.setdefault("hh_init", "zeros")
		self.kwargs.setdefault("use_low_pass_filter", False)
		self.kwargs.setdefault("low_pass_filter_alpha", np.exp(-self.dt / self.kwargs["tau_mem"]))
	
	def initialize_weights_(self):
		super().initialize_weights_()
		weight_scale = 0.2
		if "forward_weights" in self.kwargs:
			self.forward_weights.data = to_tensor(self.kwargs["forward_weights"]).to(self.device)
		else:
			torch.nn.init.normal_(self.forward_weights, mean=0.0, std=weight_scale / np.sqrt(int(self.input_size)))
		
		if "recurrent_weights" in self.kwargs and self.use_recurrent_connection:
			self.recurrent_weights.data = to_tensor(self.kwargs["recurrent_weights"]).to(self.device)
		elif self.use_recurrent_connection:
			torch.nn.init.normal_(self.recurrent_weights, mean=0.0, std=weight_scale / np.sqrt(int(self.output_size)))
	
	def create_empty_state(
			self,
			batch_size: int = 1,
			**kwargs
	) -> Tuple[torch.Tensor, ...]:
		"""
		Create an empty state in the following form:
			([membrane potential of shape (batch_size, self.output_size)],
			[synaptic current of shape (batch_size, self.output_size)],
			[spikes of shape (batch_size, self.output_size)])

		:param batch_size: The size of the current batch.
		:return: The current state.
		"""
		kwargs.setdefault("n_hh", 3 + int(self._use_low_pass_filter))
		thr = self.threshold.detach().cpu().item()
		# TODO: add the low pass filter version of Z.
		if self.kwargs["hh_init"] == "random":
			V_mu, V_std = self.kwargs.get("hh_init_mu", thr / 2.0), self.kwargs.get("hh_init_std", 0.341 * thr)
			gen = torch.Generator(device=self.device)
			gen.manual_seed(format_pseudo_rn_seed(self.kwargs.get("hh_init_seed", None)))
			V = torch.clamp_min(
				torch.rand(
					(batch_size, int(self.output_size)),
					device=self.device,
					dtype=torch.float32,
					requires_grad=True,
					generator=gen,
				) * V_std + V_mu, min=0.0
				)
			I = torch.rand(
				(batch_size, int(self.output_size)),
				device=self.device,
				dtype=torch.float32,
				requires_grad=True,
				generator=gen,
			)
			Z = self.spike_func.apply(V, self.threshold, self.gamma)
			V = V * (1.0 - Z)
			if self._use_low_pass_filter:
				Z_filtered = Z.clone()
				return tuple([V, I, Z, Z_filtered])
			else:
				return tuple([V, I, Z])
		elif self.kwargs["hh_init"] == "inputs":
			assert "inputs" in kwargs, "The inputs must be provided to initialize the state."
			assert int(self.input_size) == int(self.output_size), \
				"The input and output size must be the same with inputs initialization."
			# V_mu, V_std = self.kwargs.get("hh_init_mu", thr / 2.0), self.kwargs.get("hh_init_std", 0.341 * thr)
			V_mu, V_std = self.kwargs.get("hh_init_mu", 0.0), self.kwargs.get("hh_init_std", thr)
			gen = torch.Generator(device=self.device)
			gen.manual_seed(format_pseudo_rn_seed(self.kwargs.get("hh_init_seed", None)))
			I = torch.rand(
				(batch_size, int(self.output_size)),
				device=self.device,
				dtype=torch.float32,
				requires_grad=True,
				generator=gen,
			)
			Z = kwargs["inputs"].clone()
			V = (torch.rand(
				(batch_size, int(self.output_size)),
				device=self.device,
				dtype=torch.float32,
				requires_grad=True,
				generator=gen,
			) * V_std + V_mu)
			V = (self.beta * V + self.alpha * I) * (1.0 - Z)
			if self._use_low_pass_filter:
				Z_filtered = Z.clone()
				return tuple([V, I, Z, Z_filtered])
			else:
				return tuple([V, I, Z])
		return super(SpyLIFLayerLowPassFilter, self).create_empty_state(batch_size=batch_size, **kwargs)
	
	def forward(
			self,
			inputs: torch.Tensor,
			state: Tuple[torch.Tensor, ...] = None,
			**kwargs
	):
		assert inputs.ndim == 2, f"Inputs must be of shape (batch_size, input_size), got {inputs.shape}."
		batch_size, nb_features = inputs.shape
		if self._use_low_pass_filter:
			V, I_syn, Z, z_filtered = self._init_forward_state(state, batch_size, inputs=inputs)
		else:
			V, I_syn, Z = self._init_forward_state(state, batch_size, inputs=inputs)
			z_filtered = None
		input_current = torch.matmul(inputs, self.forward_weights)
		if self.use_recurrent_connection:
			rec_current = torch.matmul(Z, torch.mul(self.recurrent_weights, self.rec_mask))
		else:
			rec_current = 0.0
		next_I_syn = self.alpha * I_syn + input_current + rec_current
		next_V = (self.beta * V + next_I_syn) * (1.0 - Z.detach())
		next_Z = self.spike_func.apply(next_V, self.threshold, self.gamma)
		
		if self._use_low_pass_filter:
			next_z_filtered = self._low_pass_filter_alpha * z_filtered + next_Z
			return next_z_filtered, (next_V, next_I_syn, next_Z, next_z_filtered)
		else:
			return next_Z, (next_V, next_I_syn, next_Z)
		
	def extra_repr(self) -> str:
		_repr = super(SpyLIFLayerLowPassFilter, self).extra_repr()
		if self._use_low_pass_filter:
			_repr += f", low_pass_filter_alpha={self._low_pass_filter_alpha:.2f}"
		return _repr


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
	lif_layer = SpyLIFLayerLowPassFilter(
		x.shape[-1], params["n_aux_units"],
		dt=params["dt"],
		hh_init=params["hh_init"],
		force_dale_law=params["force_dale_law"],
		use_recurrent_connection=params["use_recurrent_connection"],
		use_low_pass_filter=True,
		# low_pass_filter_alpha=0.01,
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
		checkpoint_folder="./checkpoints_snn_e_prop",
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
		"W"                   : nt.to_numpy(lif_layer.forward_weights),
		"sign0"               : sign0,
		"W0"                  : W0,
		"ratio_0"             : ratio_sign_0,
		"x_pred"              : nt.to_numpy(torch.squeeze(x_pred).T),
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
			"filename"                      : None,
			"smoothing_sigma"               : 5.0,
			"n_units"                       : 500,
			"n_aux_units"                   : 500,
			"n_time_steps"                  : -1,
			"dataset_length"                : 1,
			"dataset_randomize_indexes"     : False,
			"force_dale_law"                : False,
			"learn_mu"                      : True,
			"learn_r"                       : True,
			"learn_tau"                     : True,
			"use_recurrent_connection"      : False,
		},
		n_iterations=2000,
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
		filename=f"figures/timeseries_comparison_report.png",
		show=True,
		dpi=600,
	)

	fig, axes = plt.subplots(1, 2, figsize=(12, 8))
	viz_kmeans = VisualiseKMeans(
		res["original_time_series"].T,
		nt.Size(
			[
				nt.Dimension(None, nt.DimensionProperty.NONE, "Neuron [-]"),
				nt.Dimension(None, nt.DimensionProperty.TIME, "Time [s]")]
		)
	)
	viz_kmeans.heatmap(fig=fig, ax=axes[0], title="True time series")
	Visualise(
		viz_kmeans._permute_timeseries(res["x_pred"]),
		nt.Size(
			[
				nt.Dimension(None, nt.DimensionProperty.NONE, "Neuron [-]"),
				nt.Dimension(None, nt.DimensionProperty.TIME, "Time [s]")]
		)
	).heatmap(fig=fig, ax=axes[1], title="Predicted time series")
	plt.savefig("figures/heatmap.png")
	plt.show()
	Visualise(
		res["x_pred"],
		nt.Size(
			[
				nt.Dimension(None, nt.DimensionProperty.NONE, "Neuron [-]"),
				nt.Dimension(None, nt.DimensionProperty.TIME, "Time [s]")
			]
		)
	).animate(time_interval=0.1, forward_weights=res["W"], dt=0.1, show=False, filename="figures/animation.mp4")

