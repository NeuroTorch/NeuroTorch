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
		n_neurons, n_shape = ts.shape
		sample = np.random.randint(n_neurons, size=sample_size)
		data = ts[sample, :]

		for neuron in range(data.shape[0]):
			data[neuron, :] = gaussian_filter1d(data[neuron, :], sigma=smoothing_sigma)
			data[neuron, :] = data[neuron, :] - np.min(data[neuron, :])
			data[neuron, :] = data[neuron, :] / np.max(data[neuron, :])
		self.original_time_series = data
		self.x = torch.tensor(data.T, dtype=torch.float32, device=device)

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

	@property
	def full_time_series(self):
		return self.x[None, :, :]

	@property
	def original_series(self):
		return self.original_time_series
