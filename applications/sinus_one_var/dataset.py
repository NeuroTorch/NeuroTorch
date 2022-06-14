import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor


class SinusOneVarDataset(Dataset):
	def __init__(
			self,
			*,
			n_samples: int = 1_000,
			n_steps: int = 100,
			noise_std: float = 0.0,
	):
		super().__init__()
		self.n_samples = n_samples
		self.n_steps = n_steps
		self.noise_std = noise_std

	def __len__(self):
		return self.n_samples

	def __getitem__(self, index):
		x = self.get_x_from_index(index)
		y = torch.sin(x) + torch.randn(self.n_steps) * self.noise_std
		return y

	def get_x_from_index(self, index):
		return torch.linspace(start=index, end=index+1, steps=self.n_steps)

	def show(self):
		plt.figure(figsize=(10, 5))
		for i in range(self.n_samples):
			x = self.get_x_from_index(i)
			y = self[i]
			plt.scatter(x.numpy(), y.numpy(), color='r')
		plt.show()


def get_dataloaders(
		*,
		batch_size: int = 64,
		train_val_split_ratio: float = 0.85,
		n_steps: int = 100,
		nb_workers: int = 0,
):
	"""

	:param batch_size:
	:param train_val_split_ratio:
	:param n_steps:
	:param nb_workers:
	:return:
	"""
	list_of_transform = [
		ToTensor(),
		torch.flatten,
	]

