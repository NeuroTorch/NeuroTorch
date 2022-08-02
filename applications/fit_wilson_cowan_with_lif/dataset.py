from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
from matplotlib import animation
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from neurotorch.modules.layers import WilsonCowanLayer, LearningType
from neurotorch.transforms.spikes import SpikeEncoder

from src.neurotorch.transforms.base import to_tensor


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
			spikes_transform: SpikeEncoder,
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
		self.spikes_transform = spikes_transform
		self.ts = to_tensor(self.compute_ws())
		self.t0_spikes = self.spikes_transform(torch.unsqueeze(self.ts[0], 0))
		
	def __len__(self):
		return 1
	
	def __getitem__(self, item):
		return torch.unsqueeze(self.t0_spikes, 0), self.ts[1:]

	def compute_ws(self) -> numpy.array:
		timeseries = np.zeros((self.n_steps, self.n_units))
		timeseries[0] = self.t_0
		for i in range(1, self.n_steps):
			timeseries[i] = self.layer(to_tensor(timeseries[i - 1]))[0].detach().cpu().numpy()
		return timeseries

	def plot_timeseries(self, show_matrix: bool = False):
		"""
		Plot the time series.
		"""
		if show_matrix:
			plt.imshow(self.forward_weights, cmap="RdBu_r")
			plt.colorbar()
			plt.show()
		timeseries = self.compute_ws()
		time = np.linspace(0, self.n_steps * self.dt, self.n_steps)
		plt.plot(time.T, timeseries.T)
		plt.xlabel('Time')
		plt.ylabel('Neuronal activity')
		plt.ylim([0, 1])
		plt.show()


def get_dataloader(
		*args,
		**kwargs
):
	data_loader = DataLoader(WilsonCowanTimeSeries(*args, **kwargs), batch_size=1, shuffle=False)
	return data_loader
