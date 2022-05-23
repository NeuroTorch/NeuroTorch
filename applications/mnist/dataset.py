import enum
import os

import numpy as np
import torch
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor

from neurotorch.transforms.vision import ToSpikes


class DatasetId(enum.Enum):
	MNIST = enum.auto()
	FASHION_MNIST = enum.auto()


def get_dataloaders(
		dataset_id: DatasetId,
		batch_size: int = 64,
		train_val_split_ratio: float = 0.85,
		as_timeseries: bool = True,
		n_steps: int = 100,
		to_spikes_use_periods: bool = False,
		nb_workers: int = 0,
):
	"""

	:param dataset_id:
	:param batch_size:
	:param train_val_split_ratio: The ratio of train data (i.e. train_length/data_length).
	:param as_timeseries:
	:param n_steps:
	:param to_spikes_use_periods:
	:param nb_workers:
	:return:
	"""
	list_of_transform = [
		ToTensor(),
		torch.flatten,
	]
	if as_timeseries:
		list_of_transform.append(ToSpikes(n_steps=n_steps, use_periods=to_spikes_use_periods))
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



