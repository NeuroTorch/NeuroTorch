from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy
import numpy as np
import torch
from matplotlib import animation
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from neurotorch.modules.layers import WilsonCowanLayer, LearningType

from src.neurotorch.transforms.base import to_tensor


class WilsonCowanTimeSeries:
	"""
	This class can be used to generate fake data from the initial conditions and the forward weights.
	It can also be used to predict and/or plot the time series if the initial conditions and the
	forward weights are known. To have more informations about the Wilson-Cowan dynamics, please refer
	to the documentation in layers.py -> WilsonCowanLayer class.
	"""

	def __init__(
			self,
			num_step: int,
			dt: float,
			t_0: numpy.array,
			forward_weights: numpy.array,
			mu: numpy.array or float = 0.0,
			r: numpy.array or float = 0.0,
			tau: float = 1.0
	):
		"""
		:param num_step: Number of time step in our time series
		:param dt: Time step
		:param t_0: Initial condition. array of size (number of neuron, )
		:param forward_weights: Weight matrix of size (number of neurons, number of neurons)
		:param mu: Activation threshold (number of neurons, )
		:param r: Transition rate of the RNN unit (number of neurons, )
		:param tau: Decay constant of RNN unit
		"""
		self.num_step = num_step
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
		# self.layer.forward_weights = to_tensor(self.forward_weights)
		# self.layer.mu = to_tensor(self.mu)
		# self.layer.r = to_tensor(self.r)

	@staticmethod
	def _sigmoid(x: numpy.array) -> numpy.array:
		"""
		Sigmoid function
		:return: Sigmoid of x
		"""
		return 1 / (1 + np.exp(-x))

	def _dydx(self, input: numpy.array) -> numpy.array:
		"""
		input: array of size (number of neurons, ) -> neuronal activty at a time t
		Differential equation with format dydx = f(y, x)
		Here, we have f(t, input).
		"""
		return (-input + (1 - self.r * input) * self._sigmoid(self.forward_weights @ input - self.mu)) / self.tau

	def compute_timeseries(self, transpose: bool = False) -> numpy.array:
		"""
		Compute a time series using Runge-Kutta of fourth order. The time series is compute
		from the initial condition t_0 and the forward weights.
		:param transpose: If True, the time series is returned in a column vector. Needs to be if you want to use
		the time series directly in the WilsonCowanLayer. It is automatically transpose in the get_dataset.
		"""
		num_neurons = self.t_0.shape[0]
		timeseries = np.zeros((num_neurons, self.num_step))
		timeseries[:, 0] = self.t_0
		for i in range(1, self.num_step):
			input = timeseries[:, i - 1]
			k1 = self.dt * self._dydx(input)
			k2 = self.dt * self._dydx(input + k1 / 2)
			k3 = self.dt * self._dydx(input + k2 / 2)
			k4 = self.dt * self._dydx(input + k3)
			timeseries[:, i] = input + (k1 + 2 * k2 + 2 * k3 + k4) / 6

		if transpose:
			return timeseries.T
		return timeseries

	def compute(self, transpose: bool = False) -> numpy.array:
		num_neuron = self.t_0.shape[0]
		timeseries = np.zeros((num_neuron, self.num_step))
		timeseries[:, 0] = self.t_0
		for i in range(1, self.num_step):
			timeseries[:, i] = self.layer(to_tensor(timeseries[:, i - 1]))[0].detach().cpu().numpy()
		if transpose:
			return timeseries.T
		return timeseries


	def plot_timeseries(self, show_matrix: bool = False):
		"""
		Plot the time series.
		"""
		if show_matrix:
			plt.imshow(self.forward_weights, cmap="RdBu_r")
			plt.colorbar()
			plt.show()
		timeseries = self.compute_timeseries()
		time = np.linspace(0, self.num_step * self.dt, self.num_step)
		plt.plot(time.T, timeseries.T)
		plt.xlabel('Time')
		plt.ylabel('Neuronal activity')
		plt.ylim([0, 1])
		plt.show()

	def animate_timeseries(self, step: int = 4, time_interval: float = 1.0, node_size: float = 50, alpha: float = 0.01):
		"""
		Animate the time series. The position of the nodes are obtained using the spring layout.
		Spring-Layout use the Fruchterman-Reingold force-directed algorithm. For more information,
		please refer to the following documentation of networkx:
		https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html

		:param step: Number of time step between two animation frames.
			example: if step = 4, the animation will play at t = 0, t = 4, t = 8, t = 12 ...
		:param time_interval: Time interval between two animation frames (in milliseconds)
		:param node_size: Size of the nodes
		:param alpha: Density of the connections. Small network should have a higher alpha value.
		"""
		num_frames = int(self.num_step / step)
		timeseries = self.compute_timeseries()
		connectome = nx.from_numpy_array(self.forward_weights)
		pos = nx.spring_layout(connectome)
		fig, ax = plt.subplots(figsize=(7, 7))
		nx.draw_networkx_nodes(connectome, pos, ax=ax, node_size=node_size, node_color=timeseries[:, 0], cmap="hot")
		nx.draw_networkx_edges(connectome, pos, ax=ax, width=1.0, alpha=alpha)
		x, y = ax.get_xlim()[0], ax.get_ylim()[1]
		plt.axis("off")
		text = ax.text(0, 1.15, rf"$t = 0 / {self.num_step * self.dt}$", ha="center")
		plt.tight_layout(pad=0)

		def _animate(i):
			nodes = nx.draw_networkx_nodes(connectome, pos, ax=ax, node_size=node_size,
										   node_color=timeseries[:, i * step], cmap="hot")
			text.set_text(rf"$t = {i * step * self.dt:.3f} / {self.num_step * self.dt}$")
			return nodes, text

		anim = animation.FuncAnimation(fig, _animate, frames=num_frames, interval=time_interval, blit=True)
		plt.show()


class WilsonCowanDataset(Dataset):
	"""
	Dataset for the Wilson-Cowan model.
	"""

	def __init__(self,
				 time_series: numpy.array,
				 chunk_size: int,
				 ratio: float = 0.5,
				 transform: Optional[Callable] = to_tensor
				 ):
		"""
		:param time_series: The time series of the Wilson-Cowan model. Must use shape (num_neuron, step)
		:param chunk_size: A chunk is a sequence of time series of length chunk_size.
		:param ratio: Ratio of the number of training and testing data. Always between 0 and 1 (non-inclusive).
		:param transform: Transform the data before returning it.
		"""
		super().__init__()
		self.timeseries = time_series
		self.num_step = time_series.shape[1]
		self.chunk_size = chunk_size
		if ratio <= 0 or ratio >= 1:
			raise ValueError("ratio should be between 0 and 1 (non-inclusive)")
		if chunk_size < 2:
			raise ValueError("chunk_size should be 2 or greater")
		if chunk_size > self.num_step:
			raise ValueError("chunk_size should be less than or equal to the number of time steps")
		self.ratio = ratio
		self.transform = transform

	def __len__(self) -> int:
		"""
		:return: The maximal time shift of a chunk starting from index 0
		"""
		return self.num_step - self.chunk_size

	def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Separate the data that will be use for training (x) and testing (y)
		:param index: index of the first time step of the chunk
		:returns The transpose of x and y. We need the transpose for the WilsonCowanLayer (to include the batch_size)
		"""
		self.timeseries = self.transform(self.timeseries)
		chunk = self.timeseries[:, index:index + self.chunk_size]
		x = chunk[:, : int(self.ratio * self.chunk_size)]
		y = chunk[:, int(self.ratio * self.chunk_size):]
		return torch.transpose(x, 0, 1), torch.transpose(y, 0, 1)


class WilsonCowanDataset_reproduction(Dataset):
	"""
	Dataset for the Wilson-Cowan model.
	"""

	def __init__(
			self,
			time_series: numpy.array,
			transform: Optional[Callable] = to_tensor
		):
		"""
		:param time_series: The time series of the Wilson-Cowan model. Must use shape (num_neuron, step)
		:param chunk_size: A chunk is a sequence of time series of length chunk_size.
		:param ratio: Ratio of the number of training and testing data. Always between 0 and 1 (non-inclusive).
		:param transform: Transform the data before returning it.
		"""
		super().__init__()
		self.timeseries = time_series
		self.num_step = time_series.shape[1]
		self.transform = transform
		self.t_0 = self.timeseries[:, 0][:, np.newaxis]

	def __len__(self) -> int:
		"""
		:return: The maximal time shift of a chunk starting from index 0
		"""
		return 1

	def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Separate the data that will be use for training (x) and testing (y)
		:param index: index of the first time step of the chunk
		:returns The transpose of x and y. We need the transpose for the WilsonCowanLayer (to include the batch_size)
		"""
		self.timeseries = self.transform(self.timeseries)
		self.t_0 = self.transform(self.t_0)
		x = self.t_0
		y = self.timeseries[:, 1:]
		return torch.transpose(x, 0, 1), torch.transpose(y, 0, 1)

class WSDataset(Dataset):
	def __init__(self, x):
		self.x = x

	def __len__(self):
		return 1

	def __getitem__(self, item):
		return torch.unsqueeze(self.x[0], dim=0), self.x[1:]


def get_dataloaders_reproduction(
		time_series: numpy.array,
):
	"""
	Get a dataloader for the Wilson-Cowan model.
	:param time_series: The time series of the Wilson-Cowan model.
	:return:
	"""
	train_dataset = WilsonCowanDataset_reproduction(
		time_series=time_series,
	)
	train_dataloader = DataLoader(
		train_dataset, batch_size=1, shuffle=False, num_workers=0
	)
	return dict(train=train_dataloader, val=train_dataloader, test=train_dataloader)


def get_dataloaders(
		time_series: numpy.array,
		*,
		batch_size: int = 32,
		train_val_split_ratio: float = 0.85,
		chunk_size: int = 200,
		ratio: float = 0.5,
		nb_workers: int = 0
):
	"""
	Get a dataloader for the Wilson-Cowan model.
	:param time_series: The time series of the Wilson-Cowan model.
	:param batch_size: The size of the batch
	:param train_val_split_ratio: Ratio of the number of training and testing data.
	:param chunk_size: A chunk is a sequence of time series of length
	:param ratio: Ratio of the number of training and testing data. Always between 0 and 1 (non-inclusive).
	:param nb_workers: Number of workers.
	:return:
	"""
	train_dataset = WilsonCowanDataset(
		time_series=time_series,
		chunk_size=chunk_size,
		ratio=ratio,
	)
	test_dataset = WilsonCowanDataset(
		time_series=time_series,
		chunk_size=chunk_size,
		ratio=ratio,
	)
	if np.isclose(train_val_split_ratio, 1.0):
		train_set = train_dataset
		val_dataloader = None
	else:
		train_length = int(len(train_dataset) * train_val_split_ratio)
		val_length = len(train_dataset) - train_length
		train_set, val_set = torch.utils.data.random_split(train_dataset, [train_length, val_length])
		val_dataloader = DataLoader(
			val_set, batch_size=batch_size, shuffle=False, num_workers=nb_workers
		)

	train_dataloader = DataLoader(
		train_set, batch_size=batch_size, shuffle=True, num_workers=nb_workers
	)
	test_dataloader = DataLoader(
		test_dataset, batch_size=batch_size, shuffle=False, num_workers=nb_workers
	)
	return dict(train=train_dataloader, val=val_dataloader, test=test_dataloader)