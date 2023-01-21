from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

import neurotorch as nt
from neurotorch import to_numpy, to_tensor
from typing import *
from neurotorch.modules import layers
from torch.utils.data import Dataset
from neurotorch.dimension import SizeTypes
from neurotorch.metrics import PVarianceLoss
from neurotorch.modules.layers import Linear, LILayer, WilsonCowanLayer, SpyLILayer, BellecLIFLayer, WilsonCowanLayerDebug
from neurotorch.utils import (
	unpack_out_hh,
	legend_without_duplicate_labels_,
	batchwise_temporal_recursive_filter,
	filter_parameters,
	zero_grad_params,
	recursive_detach,
)


def dz_dw_local(z: torch.Tensor, params: Sequence[torch.nn.Parameter]):
	grad_local = []
	with torch.no_grad():
		for param_idx, param in enumerate(filter_parameters(params, requires_grad=True)):
			grad_local.append(torch.zeros_like(param))
			if param.ndim >= 1:
				N_out = param.shape[-1]
			else:
				N_out = param.numel()
			# for unit_idx in range(N_out):
			# 	if param.ndim >= 1:
			# 		grad_local[param_idx][..., unit_idx] = torch.autograd.grad(
			# 			z[..., unit_idx], param,
			# 			grad_outputs=torch.ones_like(z[..., -1]),
			# 			retain_graph=True,
			# 		)[0][..., unit_idx]
			# 	else:
			# 		grad_local[param_idx] = torch.squeeze(torch.autograd.grad(z[..., unit_idx], param, retain_graph=True)[0])
			grad_local[param_idx] = torch.autograd.grad(
				z, param,
				grad_outputs=torch.ones_like(z),
				retain_graph=True,
				# allow_unused=True,
			)[0]
	return grad_local


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
		self.true_time_series = to_tensor(true_time_series)
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

		self.param_groups = [
			{"params": self.params, "lr": self.kwargs.get("params_lr", 1e-4)},
			{"params": self.output_params, "lr": self.kwargs.get("output_params_lr", 2e-4)},
		]
		# self.optimizer = torch.optim.SGD(self.param_groups, lr=0.1, maximize=True)
		# self.optimizer = torch.optim.Adam(self.param_groups, maximize=True)
		self.optimizer = torch.optim.Adam(self.param_groups)

	def _set_default_kwargs(self):
		"""
		TODO : Implement a better way to generate the random B matrix (maybe with the init...)
		"""
		self.kwargs.setdefault("kappa", 0.0)

	def _set_default_random_matrix(self):
		"""
		TODO : COPY LINEAR LAYER INIT FOR RANDOM MATRIX ... NO IDEA HOW TO DO IT
		"""
		for param_idx, param in enumerate(self.params):
			if param.ndim == 0:
				param = torch.unsqueeze(param.detach(), dim=0)
			rn_matrix = torch.rand(
				(param.shape[-1], self.true_time_series.shape[-1]),
				dtype=param.dtype, device=param.device
			).detach()
			self.random_matrices.append(rn_matrix)

	def begin(self):
		self._set_default_random_matrix()
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
		mse_func = torch.nn.MSELoss()
		self.optimizer.zero_grad()
		zero_grad_params(self.params)
		zero_grad_params(self.output_params)
		for _ in progress_bar:
			x_pred = []
			x_pred.append(self.true_time_series[:, 0, :].clone())
			forward_tensor = self.true_time_series[:, 0, :].clone().to(reservoir.device)
			hh = None

			for t in range(1, self.true_time_series.shape[-2]):
				forward_tensor, hh = unpack_out_hh(reservoir(forward_tensor, hh))
				forward_tensor, _ = unpack_out_hh(output_layer(forward_tensor, None))
				x_pred.append(forward_tensor)
				eligibility_traces = dz_dw_local(z=forward_tensor, params=self.params)
				loss_at_t = forward_tensor - self.true_time_series[:, t].to(forward_tensor.device)
				mse_at_t = mse_func(forward_tensor, self.true_time_series[:, t].to(forward_tensor.device))
				learning_signals = self.compute_learning_signals(loss_at_t)
				self.update_instantaneous_grad(mse_at_t, learning_signals, eligibility_traces)
				forward_tensor.detach_()
				hh = recursive_detach(hh)
				if t % self.update_each == 0:
					self.optimizer.step()
					self.optimizer.zero_grad()
			# x_pred_tensor = torch.stack(x_pred[1:], dim=1)
			# pvar_loss = PVarianceLoss()(x_pred_tensor, self.true_time_series[:, 1:].to(x_pred_tensor.device))
			# self.optimizer.zero_grad()
			# pvar_loss.backward()
			# for out_param in self.output_params:
			# 	out_param.grad = torch.autograd.grad(pvar_loss, out_param, retain_graph=True)[0]
			self.optimizer.step()
			self.optimizer.zero_grad()

			x_pred = torch.stack([t.cpu() for t in x_pred], dim=1)
			pvar = PVarianceLoss()(x_pred, self.true_time_series.to(x_pred.device))
			mse = torch.nn.MSELoss()(x_pred, self.true_time_series.to(x_pred.device))
			progress_bar.set_postfix({"pvar": to_numpy(pvar).item(), "MSE": to_numpy(mse).item()})
			pvars.append(to_numpy(pvar).item())
			mses.append(to_numpy(mse).item())

		return x_pred

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
			iteration=300,
			sigma=20,
			kappa=0,
			n_time_steps=-1,
			device=torch.device("cpu"),
	)
