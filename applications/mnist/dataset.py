import enum
import os
from typing import Callable

import numpy as np
import torch
import torchvision
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor

from neurotorch.transforms import LinearRateToSpikes
from neurotorch.transforms.vision import ImgToSpikes


class DatasetId(enum.Enum):
	MNIST = 0
	FASHION_MNIST = 1


def get_dataloaders(
		dataset_id: DatasetId,
		batch_size: int = 64,
		train_val_split_ratio: float = 0.85,
		# as_timeseries: bool = True,
		# n_steps: int = 100,
		# to_spikes_use_periods: bool = False,
		# inputs_linear: bool = False,
		input_transform: Callable = None,
		nb_workers: int = 0,
):
	"""

	:param dataset_id: The dataset to use.
	:param batch_size: The batch size.
	:param train_val_split_ratio: The ratio of train data (i.e. train_length/data_length).
	:param as_timeseries: Whether to use the data as time series.
	:param n_steps: The number of steps to use.
	:param to_spikes_use_periods: Whether to use the periods in the ImgToSpikes transform.
	:param nb_workers: The number of workers to use.
	:return: The dataloaders.
	"""
	list_of_transform = [
		ToTensor(),
		torchvision.transforms.Normalize((0.1307,), (0.3081,)),
	]
	# if as_timeseries:
	# 	list_of_transform.append(
	# 		LinearRateToSpikes(n_steps=n_steps)
	# 		if inputs_linear else ImgToSpikes(n_steps=n_steps, use_periods=to_spikes_use_periods)
	# 	)
	if input_transform is not None:
		list_of_transform.append(input_transform)
	transform = Compose(list_of_transform)

	if dataset_id == DatasetId.MNIST:
		root = os.path.expanduser("./data/datasets/torch/mnist")
		train_dataset = MNIST(root, train=True, download=True, transform=transform)
		test_dataset = MNIST(root, train=False, download=True, transform=transform)
	elif dataset_id == DatasetId.FASHION_MNIST:
		root = os.path.expanduser("./data/datasets/torch/fashion-mnist")
		train_dataset = FashionMNIST(root, train=True, transform=transform, download=True)
		test_dataset = FashionMNIST(root, train=False, transform=transform, download=True)
	else:
		raise ValueError()

	train_length = int(len(train_dataset) * train_val_split_ratio)
	val_length = len(train_dataset) - train_length
	train_set, val_set = torch.utils.data.random_split(train_dataset, [train_length, val_length])

	train_dataloader = DataLoader(
		train_set, batch_size=batch_size, shuffle=True, num_workers=nb_workers
	)
	val_dataloader = DataLoader(
		val_set, batch_size=batch_size, shuffle=False, num_workers=nb_workers
	)
	test_dataloader = DataLoader(
		test_dataset, batch_size=batch_size, shuffle=False, num_workers=nb_workers
	)
	return dict(train=train_dataloader, val=val_dataloader, test=test_dataloader)



