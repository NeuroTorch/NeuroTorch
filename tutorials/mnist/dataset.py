import enum
import os
from typing import Callable

import torch
import torchvision
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor


class DatasetId(enum.Enum):
	MNIST = 0
	FASHION_MNIST = 1


def get_dataloaders(
		dataset_id: DatasetId,
		batch_size: int = 64,
		train_val_split_ratio: float = 0.85,
		input_transform: Callable = None,
		nb_workers: int = 0,
		pin_memory: bool = True,
):
	"""

	:param dataset_id: The dataset to use.
	:param batch_size: The batch size.
	:param train_val_split_ratio: The ratio of train data (i.e. train_length/data_length).
	:param input_transform: The transform to apply to the input.
	:param nb_workers: The number of workers to use.
	:param pin_memory: Whether to pin memory.
	
	:return: The dataloaders.
	"""
	list_of_transform = [
		ToTensor(),
		torchvision.transforms.Normalize((0.1307,), (0.3081,)),
	]
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
		raise ValueError(f"Unknown dataset id: {dataset_id}.")

	train_length = int(len(train_dataset) * train_val_split_ratio)
	val_length = len(train_dataset) - train_length
	train_set, val_set = torch.utils.data.random_split(train_dataset, [train_length, val_length])

	train_dataloader = DataLoader(
		train_set, batch_size=batch_size, shuffle=True, num_workers=nb_workers, pin_memory=pin_memory
	)
	val_dataloader = DataLoader(
		val_set, batch_size=batch_size, shuffle=False, num_workers=nb_workers, pin_memory=pin_memory
	)
	test_dataloader = DataLoader(
		test_dataset, batch_size=batch_size, shuffle=False, num_workers=nb_workers, pin_memory=pin_memory
	)
	return dict(train=train_dataloader, val=val_dataloader, test=test_dataloader)



