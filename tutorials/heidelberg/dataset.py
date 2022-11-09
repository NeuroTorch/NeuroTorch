import os
import shutil
import warnings
from typing import Callable, Optional, Tuple

import h5py as h5py
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
import hashlib
import gzip
import urllib.request
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlretrieve

import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor

from neurotorch.transforms import LinearRateToSpikes, to_tensor


class HeidelbergDataset(Dataset):
	"""
	Dataset from: https://compneuro.net/posts/2019-spiking-heidelberg-digits/
	"""
	ONLINE_DATASET_URL = "https://compneuro.net/datasets"
	SHD_TRAIN_FILES = "shd_train.h5.gz"
	SHD_TEST_FILES = "shd_test.h5.gz"

	@staticmethod
	def _hdf5_to_tensor(hdf5_data):
		"""
		Convert the hdf5 data to a tensor.
		:param hdf5_data: The hdf5 data.
		:return: The tensor.
		"""
		return torch.as_tensor(hdf5_data)

	def __init__(
			self,
			n_steps: int = 100,
			cache_dir: str = "./data/datasets/heidelberg/shd/",
			train: bool = True,
			transform: Optional[Callable] = None,
			target_transform: Optional[Callable] = None,
			download: bool = False,
			verbose: bool = False,
			hasher: str = 'auto',
			hash_chunk_size: int = 65535,
			as_sparse: bool = False,
	):
		"""
		Constructor for the Heidelberg dataset.
		:param cache_dir: The directory where the dataset is cached.
		:param train: True if the dataset is to be used for training False for testing.
		:param transform: The transform to be applied to the inputs.
		:param target_transform: The transform to be applied to the targets.
		:param download: True if the dataset should be downloaded.
		:param verbose: True to show progress.
		:param hasher: The hasher to use for files validation.
		:param hash_chunk_size: The size of the chunks to use for hashing.
		"""
		super().__init__()
		self.n_steps = n_steps
		sub_dir = "/train" if train else "/test"
		self._cache_dir = cache_dir + sub_dir
		self._train = train
		self._file_name = self.SHD_TRAIN_FILES if train else self.SHD_TEST_FILES
		self._file_url = self.ONLINE_DATASET_URL + "/" + self._file_name
		self._gz_file_path = os.path.join(self._cache_dir, self._file_name)
		self._hdf5_file_path = self._gz_file_path.replace('.gz', '')
		self._online_file_hash = None
		self.transform = transform
		self.target_transform = target_transform
		self.verbose = verbose
		self._as_sparse = as_sparse
		if hasher.lower() in ['auto', 'sha256']:
			self._hasher = hashlib.sha256()
		elif hasher.lower() == 'md5':
			self._hasher = hashlib.md5()
		else:
			raise ValueError('Unknown hash algorithm: ' + hasher)
		self._hash_chunk_size = hash_chunk_size
		self._reporthook_progress = None
		self.data = {}
		if download:
			self.download()
		else:
			self.load_hdf5()
	
	def __getstate__(self):
		state = self.__dict__.copy()
		del state['data']
		return state

	@property
	def n_units(self):
		return self.data["unique_units"].shape[0]

	@property
	def n_labels(self):
		return self.data['unique_labels'].shape[0]

	@property
	def n_classes(self):
		return self.n_labels

	def __len__(self):
		return len(self.data['labels'])

	def getitem_as_sparse(self, index: int):
		times = self.data["spikes"]['times'][index]
		units = self.data["spikes"]['units'][index]
		labels = self.data['labels'][index]
		time_bins = np.linspace(0, self.data['max_time'], num=self.n_steps)
		time_indexes = np.digitize(times, time_bins, right=True)
		indexes = torch.LongTensor(np.asarray([time_indexes, units]))
		sparse_ts = torch.sparse_coo_tensor(indexes, torch.ones(len(times)), torch.Size([self.n_steps, self.n_units]))

		if self.transform is not None:
			sparse_ts = self.transform(sparse_ts)
		if self.target_transform is not None:
			labels = self.target_transform(labels).long()
		return sparse_ts, labels

	def getitem_as_dense(self, index: int):
		warnings.warn(
			"getitem_as_dense is deprecated and not seems to work properly, use getitem_as_sparse instead.",
			DeprecationWarning
		)
		times = self.data["spikes"]['times'][index]
		units = self.data["spikes"]['units'][index]
		labels = self.data['labels'][index]
		time_bins = np.linspace(0, self.data['max_time'], num=self.n_steps)
		time_indexes = np.digitize(times, time_bins, right=True)
		time_series = np.zeros((self.n_steps, self.n_units), dtype=np.float32)
		for t, unit in zip(time_indexes, units):
			time_series[t, unit] = 1.0

		if self.transform is not None:
			time_series = self.transform(time_series)
		if self.target_transform is not None:
			labels = self.target_transform(labels)
		return time_series, labels

	def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
		if self._as_sparse:
			return self.getitem_as_sparse(index)
		else:
			return self.getitem_as_dense(index)

	def load_hdf5(self):
		if os.path.exists(self._hdf5_file_path):
			hdf5_file = h5py.File(self._hdf5_file_path, 'r')
			self.data['spikes'] = {key: list(hdf5_file['spikes'][key]) for key in ["times", "units"]}
			self.data['labels'] = np.asarray(list(hdf5_file['labels']))
			self.data["unique_labels"] = np.unique(
				np.concatenate([np.asarray(v).flatten() for v in hdf5_file['labels']])
			)
			self.data['unique_units'] = np.unique(
				np.concatenate([np.asarray(v).flatten() for v in hdf5_file['spikes']["units"]])
			)
			self.data['max_time'] = np.max(
				np.concatenate([np.asarray(v).flatten() for v in hdf5_file['spikes']["times"]])
			)
			hdf5_file.close()
		elif os.path.exists(self._gz_file_path):
			self._unzip_file()
			self.load_hdf5()
		else:
			raise ValueError(
				f"Neither {self._hdf5_file_path} nor {self._gz_file_path} exist. "
				f"You need to download the dataset first."
			)

	def _fetch_file_hash(self):
		"""
		Fetch the hash of the file from the online dataset.
		:return: The hash of the file.
		"""
		url = self.ONLINE_DATASET_URL + "/md5sums.txt"
		response = urllib.request.urlopen(url)
		data = response.read()
		lines = data.decode('utf-8').split("\n")
		file_hashes = {line.split()[1]: line.split()[0] for line in lines if len(line.split()) == 2}
		self._online_file_hash = file_hashes[self._file_name]
		return self._online_file_hash

	def download(self):
		"""
		Download the dataset if it is not already downloaded.
		:return: None
		"""
		self._fetch_file_hash()
		self._fetch_file()
		self._unzip_file()
		self.load_hdf5()

	def _fetch_file(self):
		"""
		Download the file containing the data of the dataset if it is not already downloaded.
		:return: The path to the downloaded file.
		"""
		if self._online_file_hash is None:
			self._fetch_file_hash()
		os.makedirs(self._cache_dir, exist_ok=True)
		download = True
		if os.path.exists(self._gz_file_path) and self._online_file_hash is not None:
			try:
				if self._validate_file():
					download = False
				else:
					if self.verbose:
						print(
							f'A local file was found, but it seems to be incomplete or outdated because the hasher does '
							f'not match the original value of {self._online_file_hash} so we will re-download the data.'
						)
			except Exception as e:
				download = not os.path.exists(self._gz_file_path)
		if download:
			return self._retrieve_file()
		else:
			return self._gz_file_path

	def _retrieve_file(self):
		"""
		Retrieve the file from the online dataset.
		:return: The path to the downloaded file.
		"""
		error_msg = 'URL fetch failure on {}: {} -- {}'
		try:
			try:
				urlretrieve(self._file_url, self._gz_file_path, reporthook=self._reporthook)
			except HTTPError as e:
				raise Exception(error_msg.format(self._file_url, e.code, e.msg))
			except URLError as e:
				raise Exception(error_msg.format(self._file_url, e.errno, e.reason))
		except (Exception, KeyboardInterrupt) as e:
			if os.path.exists(self._gz_file_path):
				os.remove(self._gz_file_path)
		finally:
			if self._reporthook_progress is not None:
				self._reporthook_progress.close()
				self._reporthook_progress = None
		return self._gz_file_path

	def _validate_file(self):
		"""
		Check if the hash of the gz file is the same as the one in the online one file.
		:return: True if the hash is the same.
		"""
		raise NotImplementedError("This method is not working yet.")
		return str(self._hash_file()) == str(self._online_file_hash)

	def _hash_file(self):
		"""
		Hashes the current gz file using the hasher.
		:return: The hash of the file.
		"""
		with open(self._gz_file_path, 'rb') as fich:
			for chunk in iter(lambda: fich.read(self._hash_chunk_size), b''):
				self._hasher.update(chunk)
		return self._hasher.hexdigest()

	def _unzip_file(self):
		"""
		Unzips the gz file.
		:return: The path of the unzipped file.
		"""
		if os.path.isfile(self._hdf5_file_path):
			unzip = os.path.getctime(self._gz_file_path) > os.path.getctime(self._hdf5_file_path)
		else:
			unzip = True
		if unzip:
			if self.verbose:
				print(f"Decompressing {self._gz_file_path}")
			with gzip.open(self._gz_file_path, 'rb') as f_in, open(self._hdf5_file_path, 'wb') as f_out:
				shutil.copyfileobj(f_in, f_out)
		return self._hdf5_file_path

	def _reporthook(self, block_index: int, block_size: int, total_size: int):
		"""
		A reporthook that prints the download progress.
		:param block_index: The index of the current block.
		:param block_size: The size of the current block.
		:param total_size: The total size of the file.
		:return: None
		"""
		if self._reporthook_progress is None:
			self._reporthook_progress = tqdm(
				total=total_size,
				unit='B',
				unit_scale=True,
				desc=f'Downloading {self._file_url}',
				disable=not self.verbose
			)
		self._reporthook_progress.update(block_size)
		if self._reporthook_progress.n >= self._reporthook_progress.total:
			self._reporthook_progress.close()
			self._reporthook_progress = None


def get_dataloaders(
		*,
		batch_size: int = 256,
		train_val_split_ratio: float = 0.85,
		n_steps: int = 100,
		as_sparse: bool = False,
		download: bool = True,
		nb_workers: int = 0,
		pin_memory: bool = False,
):
	"""
	Get the dataloaders for the dataset.
	:param batch_size: The batch size.
	:param train_val_split_ratio: The ratio of the training set to the validation set.
	:param n_steps: The number of time steps for the network.
	:param as_sparse: Whether to use sparse or dense matrices.
	:param download: Whether to download the dataset.
	:param nb_workers: The number of workers to use for the dataloaders.
	:return: The dataloaders.
	"""
	if nb_workers != 0:
		raise NotImplementedError("Multiprocessing with Heidelberg Dataset is not implemented yet.")
	list_of_transform = [
		to_tensor,
	]
	transform = Compose(list_of_transform)
	train_dataset = HeidelbergDataset(
		n_steps=n_steps,
		transform=transform,
		target_transform=to_tensor,
		train=True,
		as_sparse=as_sparse,
		download=download,
		verbose=True,
	)
	test_dataset = HeidelbergDataset(
		n_steps=n_steps,
		transform=transform,
		target_transform=to_tensor,
		train=False,
		as_sparse=as_sparse,
		download=download,
		verbose=True,
	)
	train_length = int(len(train_dataset) * train_val_split_ratio)
	val_length = len(train_dataset) - train_length
	train_set, val_set = torch.utils.data.random_split(train_dataset, [train_length, val_length])

	train_dataloader = DataLoader(
		train_set, batch_size=batch_size, shuffle=True, num_workers=nb_workers, pin_memory=pin_memory,
	)
	val_dataloader = DataLoader(
		val_set, batch_size=batch_size, shuffle=False, num_workers=nb_workers, pin_memory=pin_memory,
	)
	test_dataloader = DataLoader(
		test_dataset, batch_size=batch_size, shuffle=False, num_workers=nb_workers, pin_memory=pin_memory,
	)
	return dict(train=train_dataloader, val=val_dataloader, test=test_dataloader)


if __name__ == "__main__":
	_train_dataset_ = HeidelbergDataset(train=True, download=True, verbose=True)
	_test_dataset_ = HeidelbergDataset(train=False, download=True, verbose=True)
	
	print(f"Train dataset length: {len(_train_dataset_)}")
	print(f"Test dataset length: {len(_test_dataset_)}")
	print(f"Total dataset length: {len(_train_dataset_) + len(_test_dataset_)}")



