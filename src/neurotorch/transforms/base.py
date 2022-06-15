from typing import Any

import numpy as np
import torch


def to_tensor(x: Any, dtype=torch.float32):
	if isinstance(x, np.ndarray):
		return torch.from_numpy(x).type(dtype)
	elif isinstance(x, torch.Tensor):
		return x.type(dtype)
	elif not isinstance(x, torch.Tensor):
		return torch.tensor(x, dtype=dtype)
	raise ValueError(f"Unsupported type {type(x)}")


class LinearRateToSpikes(torch.nn.Module):
	def __init__(
			self,
			n_steps: int,
			*,
			data_min: float = 0.0,
			data_max: float = 1.0,
			epsilon: float = 1e-6,
	):
		super().__init__()
		self.n_steps = n_steps
		self.data_min = to_tensor(data_min)
		self.data_max = to_tensor(data_max)
		self.epsilon = epsilon

	def firing_periods_to_spikes(self, firing_periods: np.ndarray) -> np.ndarray:
		firing_periods = np.floor(firing_periods).astype(int)
		ones_mask = firing_periods == 1
		firing_periods[firing_periods > self.n_steps] = self.n_steps
		firing_periods[firing_periods < 1] = 1
		time_dim = np.expand_dims(np.arange(1, self.n_steps+1), axis=tuple(np.arange(firing_periods.ndim)+1))
		spikes = (time_dim % firing_periods) == 0
		spikes[0, ones_mask] = 0
		return spikes.astype(float)

	def forward(self, x):
		x = to_tensor(x)
		device = x.device
		squeeze_flag = False
		if x.dim() < 1:
			x = x.unsqueeze(0)
			squeeze_flag = True
		self.data_min = torch.minimum(x, self.data_min)
		self.data_max = torch.maximum(x, self.data_max)
		x = (x - self.data_min) / (self.data_max - self.data_min + self.epsilon)
		periods = torch.floor((1 - x) * self.n_steps).type(torch.long)
		spikes = to_tensor(self.firing_periods_to_spikes(periods.cpu().numpy()))
		if squeeze_flag:
			spikes = spikes.squeeze()
		return spikes.to(device)



