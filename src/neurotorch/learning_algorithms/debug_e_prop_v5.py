from collections import defaultdict
from functools import partial

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

import neurotorch as nt
from neurotorch import to_numpy, to_tensor, SequentialRNN
from typing import *
from neurotorch.modules import layers
from torch.utils.data import Dataset
from neurotorch.dimension import SizeTypes
from neurotorch.metrics import PVarianceLoss
from neurotorch.modules.layers import Linear, LILayer, WilsonCowanLayer, SpyLILayer, BellecLIFLayer, WilsonCowanLayerDebug
from neurotorch.trainers.trainer import CurrentTrainingState, Trainer
from neurotorch.utils import (
	unpack_out_hh,
	legend_without_duplicate_labels_,
	batchwise_temporal_recursive_filter,
	filter_parameters,
	zero_grad_params,
	recursive_detach, dy_dw_local,
)
from neurotorch import Visualise, VisualisePCA, VisualiseUMAP, VisualiseKMeans


class SimplifiedEpropFinal:

	def __init__(
			self,
			true_time_series: torch.Tensor,
			params: Optional[Sequence[torch.nn.Parameter]],
			output_params: Optional[Sequence[torch.nn.Parameter]],
			learning_rate: float = 1e-2,
			update_each: int = 1,
			device: Optional[torch.device] = torch.device("cpu"),
			**kwargs
	):
		self.raw_time_series = to_tensor(true_time_series)
		self.true_time_series = to_tensor(true_time_series)
		# self.true_time_series = self.true_time_series.repeat(1024, 1, 1)
		# self.true_time_series += torch.randn_like(self.true_time_series) * 0.1
		self.params = filter_parameters(params, requires_grad=True)
		self.output_params = filter_parameters(output_params, requires_grad=True)
		self.learning_rate = learning_rate
		self.update_each = update_each
		self.device = device
		self.kwargs = kwargs
		self._set_default_kwargs()
		self.out = {}
		#self.filtered_eligibility_trace_t = torch.zeros_like(self.model.forward_weights, dtype=self.model.forward_weights.dtype, device=self.model.forward_weights.device)
		self.loss = torch.zeros_like(self.true_time_series)

		self.update_per_iter = math.ceil(true_time_series.shape[0] // self.update_each)

		self.eligibility_trace = []
		self.last_eligibility_traces = []

		self.grad_params = []
		self.grad_output_params = []
		self.random_matrices = []
		#self.data = [defaultdict(list) for _ in self.model.parameters()]
		self.learning_signal = 0.0
		
		# self.eprop = nt.learning_algorithms.Eprop(params=self.params, output_params=self.output_params)
		self.eprop = nt.learning_algorithms.Eprop(
			backward_time_steps=100,
			optim_time_steps=100,
			criterion=torch.nn.MSELoss(),
		)
		self.param_groups = [
			{"params": self.params, "lr": self.kwargs.get("params_lr", 1e-4)},
			{"params": self.output_params, "lr": self.kwargs.get("output_params_lr", 2e-4)},
		]
		# self.optimizer = torch.optim.Adam(self.eprop.initialize_param_groups())
		# self.optimizer = self.eprop.create_default_optimizer()
		self.current_training_state = CurrentTrainingState()
		self.model = kwargs.get("model", None)
		self.format_pred_batch = partial(Trainer.format_pred_batch, self)
	
	@property
	def state(self):
		"""
		Alias for the :attr:`current_training_state` attribute.

		:return: The :attr:`current_training_state`
		"""
		return self.current_training_state
		
	def _set_default_kwargs(self):
		"""
		TODO : Implement a better way to generate the random B matrix (maybe with the init...)
		"""
		self.kwargs.setdefault("kappa", 0.0)

	def begin(self):
		self.eprop.trainer = self
		self.current_training_state = self.current_training_state.update(y_batch=self.true_time_series)
		for param in self.params:
			self.last_eligibility_traces.append(None)
		return self

	def train(self, output_layer, reservoir, iteration: int = 100):
		self.begin()
		progress_bar = tqdm(
			range(iteration),
			total=iteration,
			desc=f"Training [{self.true_time_series.shape[-1]} units, {self.true_time_series.shape[-2]} time steps]",
			unit="iteration",
		)
		pvars, mses = [], []
		x_pred = None
		self.model = SequentialRNN(
			layers=[reservoir, output_layer],
			foresight_time_steps=self.true_time_series.shape[-2],
			out_memory_size=self.true_time_series.shape[-2],
			device=reservoir.device
		).build()
		self.eprop.start(self)
		for _ in progress_bar:
			self.eprop.on_train_begin(self)
			self.eprop.on_batch_begin(self)
			inputs = self.true_time_series[:, 0, :].clone().unsqueeze(1).to(self.model.device)
			x_pred = self.model.get_prediction_trace(inputs)
			self.current_training_state = self.current_training_state.update(pred_batch=x_pred)
			self.eprop.on_batch_end(self)
			self.eprop.on_train_end(self)
			
			with torch.no_grad():
				pvar = PVarianceLoss()(x_pred, self.true_time_series.to(x_pred.device))
				mse = torch.nn.MSELoss()(x_pred, self.true_time_series.to(x_pred.device))
				progress_bar.set_postfix({"pvar": to_numpy(pvar).item(), "MSE": to_numpy(mse).item()})
				pvars.append(to_numpy(pvar).item())
				mses.append(to_numpy(mse).item())
		
		val_pvars = []
		inputs = self.raw_time_series[:, 0, :].clone().unsqueeze(1).to(self.model.device)
		for _ in range(100):
			val_x_pred = self.model.get_prediction_trace(inputs)
			pvar = PVarianceLoss()(val_x_pred, self.raw_time_series.to(val_x_pred.device))
			val_pvars.append(to_numpy(pvar).item())
		print(f"Validation PVariance: {np.mean(val_pvars):.3f}")
		return x_pred, self.raw_time_series

	def compute_learning_signals(self, error: torch.Tensor):
		learning_signals = []
		error_mean = torch.mean(error.view(-1, error.shape[-1]), dim=0)
		for rn_feedback in self.random_matrices:
			learning_signals.append(torch.matmul(error_mean, rn_feedback.T.to(error_mean.device)))
		return learning_signals

	def update_instantaneous_grad(
			self,
			error: torch.Tensor,
			learning_signals: List[torch.Tensor],
			eligibility_traces: List[torch.Tensor],
	):
		with torch.no_grad():
			for param, ls, et in zip(self.params, learning_signals, eligibility_traces):
				param.grad += (ls * et.to(ls.device)).to(param.device).view(param.shape).detach()

		mean_error = torch.mean(error)
		with torch.no_grad():
			for out_param in self.output_params:
				out_param.grad += torch.autograd.grad(mean_error, out_param, retain_graph=True)[0]

	def filter_eligibility_traces(self, current_eligibility_traces: List[torch.Tensor]):
		for curr_et, last_et in zip(current_eligibility_traces, self.last_eligibility_traces):
			pass


if __name__ == '__main__':
	import os
	from typing import Optional

	import torch
	from torch.utils.data import Dataset
	import numpy as np
	from scipy.ndimage import gaussian_filter1d

	from tutorials.util import GoogleDriveDownloader
	
	class WSDataset(Dataset):
		"""
		Generate a dataset of Wilson-Cowan time series.
		This dataset is usefull to reproduce a time series using Wilson-Cowan layers.
		"""
		ROOT_FOLDER = "data/ts/"
		FILE_ID_NAME = {
			"SampleZebrafishData_PaulDeKoninckLab_2020-12-16.npy": "1-3jgAZiNU__NxxhXub7ezAJUqDMFpMCO",
		}

		def __init__(
				self,
				filename: Optional[str] = None,
				sample_size: int = 200,
				smoothing_sigma: float = 10.0,
				device: torch.device = torch.device("cpu"),
				download: bool = True,
				**kwargs
		):
			"""
			:param filename: filename of the dataset to load. If None, download the dataset from google drive.
			:param sample_size: number of neuron to use for training
			:param smoothing_sigma: sigma for the gaussian smoothing
			:param device: device to load the dataset on
			:param download: if True, download the dataset from google drive
			"""
			self.ROOT_FOLDER = kwargs.get("root_folder", self.ROOT_FOLDER)
			if filename is None:
				filename = list(self.FILE_ID_NAME.keys())[0]
				download = True
			path = os.path.join(self.ROOT_FOLDER, filename)
			if download:
				assert filename in self.FILE_ID_NAME, \
					f"File {filename} not found in the list of available files: {list(self.FILE_ID_NAME.keys())}."
				GoogleDriveDownloader(self.FILE_ID_NAME[filename], path, skip_existing=True, verbose=False).download()
			ts = np.load(path)
			n_neurons, self.max_time_steps = ts.shape
			self.seed = kwargs.get("seed", 0)
			self.random_generator = np.random.RandomState(self.seed)
			sample = self.random_generator.randint(n_neurons, size=sample_size)
			data = ts[sample, :]

			for neuron in range(data.shape[0]):
				data[neuron, :] = gaussian_filter1d(data[neuron, :], sigma=smoothing_sigma)
				data[neuron, :] = data[neuron, :] - np.min(data[neuron, :])
				data[neuron, :] = data[neuron, :] / (np.max(data[neuron, :]) + 1e-5)
			self.original_time_series = data
			self.x = torch.tensor(data.T, dtype=torch.float32, device=device)
			self._n_time_steps = int(
				np.clip(kwargs.get("n_time_steps", self.max_time_steps), -np.inf, self.max_time_steps)
			)

		@property
		def n_time_steps(self):
			return self._n_time_steps

		@n_time_steps.setter
		def n_time_steps(self, value):
			self._n_time_steps = int(np.clip(value, -np.inf, self.max_time_steps))

		def __len__(self):
			"""
			__len__ is used to get the number of samples in the dataset. Since we are training on the entire time series,
			we only have one sample which is the entire time series hence the length is 1.
			"""
			return 1

		def __getitem__(self, item):
			"""
			return the initial condition and the time series that will be use for training.
			"""
			return torch.unsqueeze(self.x[0], dim=0), self.x[1:self.n_time_steps]

		@property
		def full_time_series(self):
			return self.x[None, :, :]

		@property
		def original_series(self):
			return self.original_time_series

	def train_with_params_eprop(
			filename: Optional[str] = None,
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
			learning_rate: float = 1e-2,
			sigma: float = 20.0,
			hh_init: str = "inputs",
			checkpoint_folder="./checkpoints",
			force_dale_law: bool = False,
			kappa: float = 0.0,
			**kwargs
	):

		dataset = WSDataset(filename=filename, sample_size=kwargs.get("n_units", 50), smoothing_sigma=sigma, device=device, n_time_steps=-1)
		if kwargs.get("n_time_steps", None) is not None:
			true_time_series = torch.squeeze(dataset.full_time_series)[:kwargs["n_time_steps"], :]
		else:
			true_time_series = torch.squeeze(dataset.full_time_series)[:, :]

		ws_layer = WilsonCowanLayer(
			true_time_series.shape[-1], true_time_series.shape[1],
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
			learn_mu=learn_mu,
			learn_r=learn_r,
			learn_tau=learn_tau,
			hh_init=hh_init,
			device=device,
			name="WilsonCowan_layer1",
			force_dale_law=force_dale_law
		).build()

		linear_layer = nt.LILayer(
			true_time_series.shape[-1], true_time_series.shape[-1],
			device=device,
			use_bias=False, kappa=0.0
		).build()
		trainer = SimplifiedEpropFinal(
			true_time_series=true_time_series.unsqueeze(dim=0),
			params=ws_layer.parameters(),
			output_params=linear_layer.parameters(),
			learning_rate=learning_rate,
			update_each=kwargs.get("update_each", 1),
			device=device,
			kappa=kappa,
		)

		res_trainer = trainer.train(
			output_layer=linear_layer,
			reservoir=ws_layer,
			iteration=kwargs.get("iteration", 1000)
		)

		return res_trainer

	n_units = 200
	forward_weights = nt.init.dale_(torch.zeros(n_units, n_units), inh_ratio=0.5, rho=0.2)

	# result = np.load("res.npy", allow_pickle=True).item()

	res = train_with_params_eprop(
			dt=0.02,
			# tau=result["tau"],
			# mu=result["mu"],
			# r=result["r"],
			learn_mu=True,
			learn_r=True,
			learn_tau=True,
			forward_weights=forward_weights,
			learning_rate=1e-3,
			update_each=1,
			n_units=n_units,
			iteration=10,
			sigma=15,
			kappa=0,
			n_time_steps=-1,
			device=torch.device("cuda"),
	)

	predicted, true = res

	predicted = torch.squeeze(predicted)
	true = torch.squeeze(true)

	pred_viz = Visualise(
		predicted,
		shape=nt.Size(
			[
				nt.Dimension(None, nt.DimensionProperty.TIME, "Time [s]"),
				nt.Dimension(None, nt.DimensionProperty.NONE, "Activity [-]"),
			]
		)
	)
	pred_viz.plot_timeseries_comparison_report(
		true,
		title=f"Prediction",
		#filename=f"{res['network'].checkpoint_folder}/figures/WilsonCowan_prediction_report.png",
		show=True,
		dpi=600,
	)
