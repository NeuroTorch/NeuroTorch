import numpy as np
import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor


class SinusSpikesDataset(Dataset):
	def __init__(
			self,
			*,
			n_samples: int = 1_000,
			threshold: float = 0.95,
			n_steps: int = 100,
			n_variables: int = 1,
			noise_std: float = 0.1,
	):
		super().__init__()
		self.n_samples = n_samples
		self.threshold = threshold
		self.n_steps = n_steps
		self.n_variables = n_variables
		self.noise_std = noise_std
		self.rn_phases = np.random.uniform(0, 2 * np.pi, n_variables)

	def __len__(self):
		return self.n_samples

	def __getitem__(self, index):
		x = self.get_x_from_index(index).unsqueeze(1)
		y = torch.sin(x + self.rn_phases) + torch.randn(self.n_steps, self.n_variables) * self.noise_std
		return (y > self.threshold).float().squeeze()

	def get_x_from_index(self, index):
		return torch.linspace(start=index, end=index+1, steps=self.n_steps)

	def show(self):
		fig, ax = plt.subplots(figsize=(10, 5))
		y = np.concatenate([self[i].numpy() for i in range(self.n_samples)])
		if y.ndim == 1:
			y = y[:, np.newaxis]
		x = np.concatenate([self.get_x_from_index(i).numpy() for i in range(self.n_samples)])
		line_length = 1.0
		pad = 0.5
		for n_idx, spikes in enumerate(y.T):
			spikes_idx = x[np.isclose(spikes, 1.0)]
			ymin = (spikes.shape[-1] - n_idx) * (pad + line_length)
			ax.vlines(spikes_idx, ymin=ymin, ymax=ymin + line_length, colors=[0, 0, 0])
		ax.get_yaxis().set_visible(False)
		plt.show()


def get_dataloaders(
		*,
		batch_size: int = 252,
		train_val_split_ratio: float = 0.85,
		n_steps: int = 100,
		n_variables: int = 1,
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
	train_dataset = SinusSpikesDataset()

