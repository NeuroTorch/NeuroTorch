import numbers
from typing import Any, Optional

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


def to_numpy(x: Any, dtype=np.float32):
	if isinstance(x, np.ndarray):
		return np.asarray(x, dtype=dtype)
	elif isinstance(x, torch.Tensor):
		return x.detach().cpu().numpy()
	elif isinstance(x, numbers.Number):
		return x
	elif isinstance(x, dict):
		return {k: to_numpy(v, dtype=dtype) for k, v in x.items()}
	elif not isinstance(x, torch.Tensor):
		return np.asarray(x, dtype=dtype)
	raise ValueError(f"Unsupported type {type(x)}")


class ToDevice(torch.nn.Module):
	def __init__(self, device: torch.device, non_blocking: bool = True):
		super().__init__()
		self.device = device
		self.non_blocking = non_blocking

	def forward(self, x: torch.Tensor):
		if x is None:
			return x
		if not isinstance(x, torch.Tensor):
			if isinstance(x, dict):
				return {k: self.forward(v) for k, v in x.items()}
			elif isinstance(x, list):
				return [self.forward(v) for v in x]
			elif isinstance(x, tuple):
				return tuple(self.forward(v) for v in x)
			else:
				return x
		return x.to(self.device, non_blocking=self.non_blocking)
	
	def __repr__(self):
		return f"ToDevice({self.device}, async={self.non_blocking})"


class ToTensor(torch.nn.Module):
	def __init__(self, dtype=torch.float32, device: Optional[torch.device] = None):
		super().__init__()
		self.dtype = dtype
		self.device = device
		if self.device is None:
			self.to_device = None
		else:
			self.to_device = ToDevice(self.device)
	
	def forward(self, x: Any) -> torch.Tensor:
		x = to_tensor(x, self.dtype)
		if self.to_device is not None:
			x = self.to_device(x)
		return x


class IdentityTransform(torch.nn.Module):
	def __init__(self):
		super().__init__()
	
	def forward(self, x: Any):
		return x


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
	
	def __repr__(self):
		return f"{self.__class__.__name__}(n_steps={self.n_steps})"

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


class ConstantValuesTransform(torch.nn.Module):
	def __init__(self, n_steps: int, batch_wise: bool = True):
		super().__init__()
		self.n_steps = n_steps
		self.batch_wise = batch_wise
	
	def __repr__(self):
		return f"{self.__class__.__name__}(n_steps={self.n_steps})"

	def forward(self, x: Any):
		x = to_tensor(x)
		if self.batch_wise:
			return x.repeat(1, self.n_steps, 1)
		return x.repeat(self.n_steps, 1)

