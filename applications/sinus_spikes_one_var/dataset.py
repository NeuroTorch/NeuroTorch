from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

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
			transform: Optional[Callable] = None,
			target_transform: Optional[Callable] = None,
	):
		super().__init__()
		self.n_samples = n_samples
		self.threshold = threshold
		self.n_steps = n_steps
		self.n_variables = n_variables
		self.noise_std = noise_std
		self.rn_phases = np.random.uniform(0, 2 * np.pi, n_variables)
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return self.n_samples

	def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
		x, x_targets = self.get_x_from_index(index)
		x, x_targets = x.unsqueeze(1), x_targets.unsqueeze(1)
		y = torch.sin(x + self.rn_phases) + torch.randn(self.n_steps, self.n_variables) * self.noise_std
		spikes = (y > self.threshold).float().squeeze()

		targets = torch.sin(x_targets + self.rn_phases) + torch.randn(self.n_steps, self.n_variables) * self.noise_std
		spikes_targets = (targets > self.threshold).float().squeeze()
		if self.transform is not None:
			spikes = self.transform(spikes)
		if self.target_transform is not None:
			spikes_targets = self.target_transform(spikes_targets)
		return spikes, spikes_targets

	def get_x_from_index(self, index):
		x = torch.linspace(0, index * np.pi, self.n_steps)
		x_targets = torch.linspace(index * np.pi, (index + 1) * np.pi, self.n_steps)
		return x, x_targets

	def show(self):
		fig, ax = plt.subplots(figsize=(10, 5))
		y = np.concatenate([self[i][0].numpy() for i in range(self.n_samples)])
		if y.ndim == 1:
			y = y[:, np.newaxis]
		x = np.concatenate([self.get_x_from_index(i)[0].numpy() for i in range(self.n_samples)])
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
	train_dataset = SinusSpikesDataset(
		n_variables=n_variables,
		n_steps=n_steps,
	)
	test_dataset = SinusSpikesDataset(
		n_variables=n_variables,
		n_steps=n_steps,
	)
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
