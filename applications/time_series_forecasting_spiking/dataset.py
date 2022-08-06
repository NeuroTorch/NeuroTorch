from typing import Callable, Optional, Tuple, Iterable

import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
from matplotlib import animation
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from scipy.ndimage import gaussian_filter1d
import neurotorch as nt
from neurotorch.modules.layers import WilsonCowanLayer, LearningType
from neurotorch.transforms.spikes_encoders import SpikesEncoder, LIFEncoder, ALIFEncoder, SpyLIFEncoder

from src.neurotorch.transforms.base import to_tensor


class TimeSeriesDataset(Dataset):
	def __init__(
			self,
			input_transform: Optional[torch.nn.Module] = None,
			target_transform: Optional[torch.nn.Module] = None,
			n_units: Optional[int] = None,
			units: Optional[Iterable[int]] = None,
			seed : int = 0
	):
		super().__init__()
		self.ts = np.load('timeSeries_2020_12_16_cr3_df.npy')
		self.n_neurons, self.n_time_steps = self.ts.shape
		
		random_generator = np.random.RandomState(seed)
		
		if units is not None:
			units = list(units)
			assert n_units is None
			n_units = len(units)
		elif n_units is not None:
			units = random_generator.randint(self.n_neurons, size=n_units)
		else:
			n_units = 128
			units = random_generator.randint(self.n_neurons, size=n_units)
		
		self.n_units = n_units
		self.units_indexes = units
		self.data = self.ts[self.units_indexes, :]
		self.sigma = 30
		
		for neuron in range(self.data.shape[0]):
			self.data[neuron, :] = gaussian_filter1d(self.data[neuron, :], sigma=self.sigma)
			self.data[neuron, :] = self.data[neuron, :] - np.min(self.data[neuron, :])
			self.data[neuron, :] = self.data[neuron, :] / np.max(self.data[neuron, :])
		
		self.data = nt.to_tensor(self.data.T, dtype=torch.float32)
		self.transform = input_transform
		self.target_transform = target_transform
		if self.transform is None:
			self.t0_transformed = torch.unsqueeze(self.data[0], dim=0)
		else:
			self.t0_transformed = self.transform(torch.unsqueeze(self.data[0], dim=0))
			
		self.target = self.data
		if self.target_transform is None:
			self.target_transformed = self.target
		else:
			self.target_transformed = self.transform(self.target)
	
	def __len__(self):
		return 1
	
	def __getitem__(self, item):
		return self.t0_transformed, self.target_transformed


class WilsonCowanTimeSeries(Dataset):
	"""
	This class can be used to generate fake data from the initial conditions and the forward weights.
	It can also be used to predict and/or plot the time series if the initial conditions and the
	forward weights are known. To have more information about the Wilson-Cowan dynamics, please refer
	to the documentation in layers.py -> WilsonCowanLayer class.
	"""

	def __init__(
			self,
			n_steps: int,
			dt: float,
			t_0: numpy.array,
			forward_weights: numpy.array,
			transform: torch.nn.Module,
			mu: numpy.array or float = 0.0,
			r: numpy.array or float = 0.0,
			tau: float = 1.0
	):
		"""
		:param n_steps: Number of time step in our time series
		:param dt: Time step
		:param t_0: Initial condition. array of size (number of neuron, )
		:param forward_weights: Weight matrix of size (number of neurons, number of neurons)
		:param mu: Activation threshold (number of neurons, )
		:param r: Transition rate of the RNN unit (number of neurons, )
		:param tau: Decay constant of RNN unit
		"""
		self.n_steps = n_steps
		self.n_units = t_0.shape[0]
		self.dt = dt
		self.t_0 = t_0
		self.forward_weights = forward_weights
		self.mu = mu
		self.r = r
		self.tau = tau
		self.layer = WilsonCowanLayer(
			input_size=self.forward_weights.shape[0],
			output_size=self.forward_weights.shape[1],
			learning_type=LearningType.NONE,
			dt=self.dt,
			forward_weights=self.forward_weights,
			mu=self.mu,
			r=self.r,
			tau=self.tau
		)
		self.layer.build()
		self.transform = transform
		self.ts = to_tensor(self.compute_ws())
		self.t0_spikes = self.transform(torch.unsqueeze(self.ts[0], 0))
		self.t_space_ms = np.linspace(0, self.n_steps * self.dt, self.n_steps) * 1000
		
	def __len__(self):
		return 1
	
	def __getitem__(self, item):
		# return self.t0_spikes, self.ts[1:]
		return self.t0_spikes, self.ts

	def compute_ws(self) -> numpy.array:
		timeseries = np.zeros((self.n_steps, self.n_units))
		timeseries[0] = self.t_0
		for i in range(1, self.n_steps):
			timeseries[i] = self.layer(to_tensor(timeseries[i - 1]))[0].detach().cpu().numpy()
		return timeseries

	def plot_timeseries(self, fig: Optional[plt.Figure] = None, axes: Optional[plt.Axes] = None, show: bool = True):
		"""
		Plot the time series.
		"""
		if fig is None or axes is None:
			fig, axes = plt.subplots(1, 2, figsize=(15, 5))
		timeseries = self.compute_ws()
		self.raster_plot(axes[0])
		axes[0].set_xlabel("Time [ms]")
		axes[0].set_ylabel("Neurons [-]")
		axes[0].set_title(f"{self.transform.__name__}: Raster plot $t_0$")
		axes[1].plot(self.t_space_ms, timeseries)
		axes[1].set_title("Wilson-Cowan Time series")
		axes[1].set_xlabel("Time [ms]")
		axes[1].set_ylabel("Neuronal activity [-]")
		axes[1].set_ylim([0, 1])
		if show:
			plt.show()
	
	def raster_plot(self, ax):
		line_length = 1.0
		pad = 0.5
		t_space = np.linspace(
			0, self.transform.n_steps * self.transform.dt, self.transform.n_steps
		) * 1000
		for n_idx, spikes in enumerate(self.t0_spikes.detach().cpu().numpy().T):
			spikes_idx = t_space[np.isclose(spikes, 1.0)]
			ymin = (self.t0_spikes.shape[-1] - n_idx) * (pad + line_length)
			ax.vlines(spikes_idx, ymin=ymin, ymax=ymin + line_length, colors=[0, 0, 0])
		# ax.get_yaxis().set_visible(False)
		ax.set_yticks([])
	
	def plot_forward_weights(self):
		"""
		Plot the forward weights.
		"""
		plt.imshow(self.forward_weights, cmap="RdBu_r")
		plt.colorbar()
		plt.show()


def get_dataloader(
		*args,
		**kwargs
):
	data_loader = DataLoader(WilsonCowanTimeSeries(*args, **kwargs), batch_size=1, shuffle=False)
	return data_loader


if __name__ == '__main__':
	_n_units_ = 10
	_dt_ = 2e-2
	t_0 = np.random.rand(_n_units_)
	forward_weights = 3 * np.random.randn(_n_units_, _n_units_)
	mu = np.random.randn(_n_units_, )
	r = np.random.rand(1).item()
	
	fig, axes = plt.subplots(3, 2, figsize=(18, 8))
	for i, (line_axes, encoder) in enumerate(zip(axes, [LIFEncoder, ALIFEncoder, SpyLIFEncoder])):
		ws = WilsonCowanTimeSeries(
			n_steps=1_000,
			dt=_dt_,
			t_0=t_0,
			forward_weights=forward_weights,
			transform=encoder(
				n_steps=32,
				n_units=_n_units_,
				dt=_dt_,
			),
			mu=mu,
			r=r,
			tau=1.0,
		)
		print(f"shape: {ws[0][0].shape, ws[0][1].shape}")
		ws.plot_timeseries(fig=fig, axes=line_axes, show=False)
	fig.tight_layout()
	plt.show()
	


