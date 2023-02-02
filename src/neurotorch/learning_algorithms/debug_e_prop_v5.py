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
			device: Optional[torch.device] = torch.device("cpu"),
			**kwargs
	):
		self.raw_time_series = to_tensor(true_time_series)
		self.true_time_series = to_tensor(true_time_series)
		self.device = device
		self.kwargs = kwargs
		
		self.eprop = nt.learning_algorithms.Eprop(
			# backward_time_steps=1,
			# optim_time_steps=1,
			# criterion=torch.nn.MSELoss(),
		)
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

	def train(self, output_layer, reservoir, iteration: int = 100):
		self.eprop.trainer = self
		self.current_training_state = self.current_training_state.update(y_batch=self.true_time_series)
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
			foresight_time_steps=self.true_time_series.shape[-2] - 1,
			out_memory_size=self.true_time_series.shape[-2] - 1,
			device=reservoir.device
		).build()
		print(self.model)
		self.eprop.start(self)
		for _ in progress_bar:
			# self.true_time_series = self.raw_time_series.clone().repeat(32, 1, 1)
			# self.true_time_series += torch.randn_like(self.true_time_series) * 0.1
			self.current_training_state = self.current_training_state.update(y_batch=self.true_time_series)
			self.eprop.on_train_begin(self)
			self.eprop.on_batch_begin(self)
			inputs = self.true_time_series[:, 0, :].clone().unsqueeze(1).to(self.model.device)
			x_pred = self.model.get_prediction_trace(inputs)
			x_pred = torch.concat([inputs, x_pred], dim=1)
			self.current_training_state = self.current_training_state.update(pred_batch=x_pred)
			self.eprop.on_batch_end(self)
			self.eprop.on_train_end(self)
			
			with torch.no_grad():
				pvar = PVarianceLoss()(x_pred, self.true_time_series.to(x_pred.device))
				mse = torch.nn.MSELoss()(x_pred, self.true_time_series.to(x_pred.device))
				val_pvar = self.validate(1)
				progress_bar.set_postfix({
					"pvar": to_numpy(pvar).item(),
					"val_pvar": val_pvar,
					"MSE": to_numpy(mse).item()
				})
				pvars.append(to_numpy(pvar).item())
				mses.append(to_numpy(mse).item())
		
		val_pvar = self.validate(100)
		print(f"Validation PVariance: {val_pvar:.3f}")
		return x_pred, self.raw_time_series

	def validate(self, n: int = 1):
		val_pvars = []
		inputs = self.raw_time_series[:, 0, :].clone().unsqueeze(1).to(self.model.device)
		for _ in range(n):
			val_x_pred = self.model.get_prediction_trace(inputs)
			val_x_pred = torch.concat([inputs, val_x_pred], dim=1)
			pvar = PVarianceLoss()(val_x_pred, self.raw_time_series.to(val_x_pred.device))
			val_pvars.append(to_numpy(pvar).item())
		return np.mean(val_pvars).item()



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
			self.original_time_series = ts.copy()
			if kwargs.get("rm_dead_units", True):
				ts = ts[np.sum(ts, axis=-1) > 0, :]
			n_neurons, self.max_time_steps = ts.shape
			self.seed = kwargs.get("seed", 0)
			self.random_generator = np.random.RandomState(self.seed)
			sample = self.random_generator.randint(n_neurons, size=sample_size)
			data = ts[sample, :]

			for neuron in range(data.shape[0]):
				data[neuron, :] = gaussian_filter1d(data[neuron, :], sigma=smoothing_sigma)
				data[neuron, :] = data[neuron, :] - np.min(data[neuron, :])
				data[neuron, :] = data[neuron, :] / (np.max(data[neuron, :]) + 1e-5)
			
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
			return to_tensor(self.original_time_series, dtype=torch.float32)

	def train_with_params_eprop(
			filename: Optional[str] = None,
			forward_weights: Optional[torch.Tensor or np.ndarray] = None,
			std_weights: float = 1,
			dt: float = 2e-2,
			mu: Optional[float or torch.Tensor or np.ndarray] = 0.0,
			mean_mu: Optional[float] = 0.0,
			std_mu: Optional[float] = 1.0,
			r: Optional[float or torch.Tensor or np.ndarray] = 0.1,
			mean_r: Optional[float] = 0.5,
			std_r: Optional[float] = 0.4,
			tau: float = 0.1,
			learn_mu: bool = True,
			learn_r: bool = True,
			learn_tau: bool = True,
			device: torch.device = torch.device("cpu"),
			learning_rate: float = 1e-2,
			sigma: float = 15.0,
			hh_init: str = "inputs",
			checkpoint_folder="./checkpoints",
			force_dale_law: bool = False,
			**kwargs
	):
		from tutorials.learning_algorithms.dataset import get_dataloader
		torch.manual_seed(0)
		old_dataset = WSDataset(filename=filename, sample_size=kwargs.get("n_units", 50), smoothing_sigma=sigma, device=device, n_time_steps=-1)
		dataloader = get_dataloader(
			batch_size=kwargs.get("batch_size", 512), verbose=True, n_workers=kwargs.get("n_workers"),
			filename=filename, n_units=kwargs.get("n_units", 50), smoothing_sigma=sigma, device=device,
			n_time_steps=-1, rm_dead_units=True
		)
		dataset = dataloader.dataset
		# assert torch.allclose(dataset.full_time_series, old_dataset.full_time_series), "The dataset is not the same as the old one."
		ws_layer = WilsonCowanLayer(
			dataset.n_units, dataset.n_units,
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

		linear_layer = nt.Linear(
			dataset.n_units, dataset.n_units,
			device=device,
			use_bias=False,
			activation="sigmoid",
		).build()
		trainer = SimplifiedEpropFinal(
			true_time_series=dataset.full_time_series,
			device=device,
		)

		res_trainer = trainer.train(
			output_layer=linear_layer,
			reservoir=ws_layer,
			iteration=kwargs.get("iteration", 1000)
		)

		return res_trainer

	n_units = 200
	# forward_weights = nt.init.dale_(torch.zeros(n_units, n_units), inh_ratio=0.5, rho=0.2)

	# result = np.load("res.npy", allow_pickle=True).item()

	res = train_with_params_eprop(
			dt=0.02,
			learn_mu=True,
			learn_r=True,
			learn_tau=True,
			# forward_weights=forward_weights,
			learning_rate=1e-3,
			n_units=n_units,
			iteration=200,
			sigma=15,
			n_time_steps=-1,
			device=torch.device("cpu"),
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
