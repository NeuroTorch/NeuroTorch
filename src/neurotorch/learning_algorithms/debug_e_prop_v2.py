import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

import neurotorch as nt
from neurotorch import to_numpy, to_tensor
from typing import *
from neurotorch.modules import layers
from neurotorch.dimension import SizeTypes
from neurotorch.metrics import PVarianceLoss
from neurotorch.modules.layers import Linear, LILayer, WilsonCowanLayer, SpyLILayer, BellecLIFLayer, WilsonCowanLayerDebug
from neurotorch.utils import unpack_out_hh, legend_without_duplicate_labels_, batchwise_temporal_recursive_filter



class SimplifiedEprop:

	def __init__(
			self,
			true_time_series: torch.Tensor,
			model: layers.BaseLayer,
			learning_rate: float = 1e-2,
			update_each: int = 1,
			iteration: int = 100,
			device: Optional[torch.device] = torch.device("cpu"),
			**kwargs
	):
		self.true_time_series = to_tensor(true_time_series)
		self.model = model
		self.learning_rate = learning_rate
		self.update_each = update_each
		self.iteration = iteration
		self.device = device
		self.kwargs = kwargs
		self._set_default_kwargs()
		self.out = {}
		self.filtered_eligibility_trace_t = torch.zeros_like(self.model.forward_weights, dtype=self.model.forward_weights.dtype, device=self.model.forward_weights.device)
		self.loss = torch.zeros_like(self.true_time_series)

		self.update_per_iter = math.ceil(true_time_series.shape[0] // self.update_each)

		self.eligibility_trace_t = []
		self.eligibility_trace_t_minus_1 = None
		self.learning_signal_with_eligibility_trace_at_t = []
		self.delta_params = []
		self.random_matrices = []

	def _set_default_kwargs(self):
		"""
		TODO : Implement a better way to generate the random B matrix (maybe with the init...)
		"""
		self.kwargs.setdefault("kappa", 0.0)
		#random_matrix = torch.randn(self.model.forward_weights.shape, dtype=self.model.forward_weights.dtype, device=self.model.forward_weights.device)
		#self.kwargs.setdefault("B", random_matrix)

	def _set_default_random_matrix(self):
		for param_idx, param in enumerate(self.model.parameters()):
			if param.ndim == 0:
				param = torch.unsqueeze(param.detach(), dim=0)
			self.random_matrices.append(torch.randn((param.shape[-1], self.true_time_series.shape[-1]), dtype=param.dtype, device=param.device).detach())

	def begin(self):
		self._set_default_random_matrix()
		for param in self.model.parameters():
			if param.ndim == 0:
				param = torch.unsqueeze(param.detach(), dim=0)
			self.delta_params.append(torch.zeros_like(param, dtype=param.dtype, device=param.device).detach())
			self.eligibility_trace_t.append(torch.zeros_like(param, dtype=param.dtype, device=param.device).detach())
			self.learning_signal_with_eligibility_trace_at_t.append(torch.zeros_like(param, dtype=param.dtype, device=param.device).detach())
		with torch.no_grad():
			self.out["true_time_series"] = self.true_time_series
			self.out["W0"] = self.model.forward_weights.clone()
			self.out["mu0"] = self.model.mu.clone()
			self.out["r0"] = self.model.r.clone()
			self.out["tau0"] = self.model.tau.clone()
			#self.out["B"] = self.kwargs["B"].clone()
			self.out["kappa"] = self.kwargs["kappa"]
			return self

	def train(self):
		self.begin()
		progress_bar = tqdm(
			range(self.iteration),
			total=self.iteration,
			desc="Training",
			unit="iteration",
		)

		for _ in progress_bar:
			x_pred = []
			x_pred.append(self.true_time_series[0].clone())
			forward_tensor = self.true_time_series[0].clone()

			for t in range(1, self.true_time_series.shape[0]):
				forward_tensor = self.model(forward_tensor)[0]
				x_pred.append(forward_tensor)
				self.compute_dz_dw_local(forward_tensor)
				loss_at_t = forward_tensor - self.true_time_series[t]
				self.compute_learning_signal_with_eligibility_trace(loss_at_t)
				self.update_delta_param()
				if t % self.update_each == 0:
					self.update_parameters()
					self.reset_parameters_update()

			if self.true_time_series.shape[0] % self.update_each != 0:
				# Final update
				self.update_parameters()
				self.reset_parameters_update()

			x_pred = torch.stack(x_pred, dim=0)
			pvar = PVarianceLoss()(x_pred, self.true_time_series)
			progress_bar.set_postfix({"pvar": pvar.detach().item()})

		self.out["x_pred"] = x_pred






			# for update_complete in range(self.update_per_iter):
			# 	x_pred = []
			# 	x_pred.append(self.true_time_series[0].clone())
			# 	forward_tensor = self.true_time_series[0].clone()
			#
			# 	for i in range(self.update_each)
			# 		forward_tensor = self.model(forward_tensor)
			# 		x_pred.append(forward_tensor)


	def compute_dz_dw_local(self, z: torch.tensor):
		"""
		Equation (13)
		"""
		for param_idx, param in enumerate(self.model.parameters()):
			# for each neuron at a time step t
			for neuron_idx in range(self.true_time_series.shape[1]):
				if param.requires_grad:
					try:
						self.eligibility_trace_t[param_idx][:, neuron_idx] = torch.autograd.grad(z[neuron_idx], param, retain_graph=True)[0][:, neuron_idx]
					except:
						self.eligibility_trace_t[param_idx][:, neuron_idx] = torch.autograd.grad(z[neuron_idx], param, retain_graph=True)[0][:, neuron_idx]


	def compute_learning_signal_with_eligibility_trace(self, loss_at_t: torch.tensor):
		"""
		Compute L_j^t * e_{ij}^t for equation (28)
		Since loss = y^t - y_pred^t where y [1xN]
		loss @ B.T = B @ loss.T -> loss.T is the biological convention where loss is the NeuroTorch convention
		"""
		# self.filtered_eligibility_trace_t()
		for param_idx, _ in enumerate(self.model.parameters()):
			learning_signal_at_t = loss_at_t @ self.random_matrices[param_idx].T
			self.learning_signal_with_eligibility_trace_at_t[param_idx] = learning_signal_at_t * self.eligibility_trace_t[param_idx]



	def filter_eligibility_trace_t(self):
		"""
		TODO : Equation (12) -> Only usefull if using spiking (now, kappa=0).
		"""
		pass

	def update_delta_param(self):
		"""
		Equation (28)
		"""
		for param_idx, param in enumerate(self.model.parameters()):
			self.delta_params[param_idx] += -self.learning_rate * self.learning_signal_with_eligibility_trace_at_t[param_idx]

	def update_parameters(self):
		with torch.no_grad():
			for param_idx, param in enumerate(self.model.parameters()):
				new_param = param + self.delta_params[param_idx]
				if param.ndim == 0:
					param.copy_(torch.squeeze(new_param))
				else:
					param.copy_(new_param)
			self.model.zero_grad()
				#self.model.parameters()[param_idx] = param + self.delta_params[param_idx]

	def reset_parameters_update(self):
		"""
		Apply this function after each update of the parameters
		"""
		for param_idx, param in enumerate(self.model.parameters()):
			if param.ndim == 0:
				param = torch.unsqueeze(param.detach(), dim=0)
			self.delta_params[param_idx] = torch.zeros_like(param, dtype=param.dtype, device=param.device).detach()
			self.eligibility_trace_t[param_idx] = torch.zeros_like(param, dtype=param.dtype, device=param.device).detach()
			self.learning_signal_with_eligibility_trace_at_t[param_idx] = (torch.zeros_like(param, dtype=param.dtype, device=param.device)).detach()


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
				np.clip(kwargs.get("n_time_steps", self.max_time_steps), -np.inf, self.max_time_steps))

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


	def train_with_params(
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
			**kwargs
	):

		dataset = WSDataset(filename=filename, sample_size=kwargs.get("n_units", 50), smoothing_sigma=sigma, device=device)
		true_time_series = torch.squeeze(dataset.full_time_series)

		ws_layer = WilsonCowanLayerDebug(
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

		trainer = SimplifiedEprop(
			true_time_series=true_time_series,
			model=ws_layer,
			learning_rate=learning_rate,
			update_each=1,
			iteration=50,
			device=device
		)
		trainer.train()

		return "done"
	n_units = 50
	forward_weights = nt.init.dale_(torch.zeros(n_units, n_units), inh_ratio=0.5, rho=0.2)

	res = train_with_params(
			dt=0.02,
			learn_mu=False,
			learn_r=False,
			learn_tau=False,
			forward_weights=forward_weights,
	)

