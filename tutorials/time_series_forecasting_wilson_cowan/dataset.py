import torch
from torch.utils.data import Dataset


class WSDataset(Dataset):
	"""
	Generate a dataset of Wilson-Cowan time series.
	This dataset is usefull to reproduce a time series using Wilson-Cowan layers.
	"""

	def __init__(self, x):
		"""
		:param x: Time series of shape (num_step, num_neurons)
		"""
		self.x = x

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
		return torch.unsqueeze(self.x[0], dim=0), self.x[1:]
