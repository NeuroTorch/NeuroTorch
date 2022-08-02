from typing import Any, Type, Union, Optional

import numpy as np
import torch

from . import ConstantValuesTransform, to_tensor
from .. import LIFLayer, SpyLIFLayer, ALIFLayer, LearningType


class SpikesEncoder(torch.nn.Module):
	def __init__(
			self,
			n_steps: int,
			n_units: int,
			spikes_layer_type: Type[Union[LIFLayer, SpyLIFLayer, ALIFLayer]],
			dt: float = 1e-3,
			device: Optional[torch.device] = None,
			spikes_layer_kwargs: Optional[dict] = None,
	):
		super().__init__()
		self.n_steps = n_steps
		self.const_transform = ConstantValuesTransform(n_steps)
		self.spikes_layer_type = spikes_layer_type
		if spikes_layer_kwargs is None:
			spikes_layer_kwargs = {}
		
		assert "forward_weights" not in spikes_layer_kwargs, \
			"forward_weights cannot be specified since it must be eye(n_units)"
		assert "learning_type" not in spikes_layer_kwargs, \
			"learning_type cannot be specified since it must be LearningType.NONE"
		assert "use_recurrent_connection" not in spikes_layer_kwargs, \
			"use_recurrent_connection cannot be specified since it must be False"
		assert "dt" not in spikes_layer_kwargs, \
			"dt cannot be specified since it must be the given dt"
		assert "device" not in spikes_layer_kwargs, \
			"device cannot be specified since it must be the given device"
		
		self.spikes_layer = self.spikes_layer_type(
			n_units, n_units,
			forward_weights=torch.eye(n_units),
			use_recurrent_connection=False,
			learning_type=LearningType.NONE,
			dt=dt,
			device=device,
			**spikes_layer_kwargs,
		)
		self.spikes_layer.build()

	def forward(self, x: Any):
		x = to_tensor(x).to(self.spikes_layer.device)
		x = self.const_transform(x)[np.newaxis, :]
		x_spikes = []
		hh = None
		for t in range(self.n_steps):
			spikes, hh = self.spikes_layer(x[:, t], hh)
			x_spikes.append(spikes)
		x_spikes = torch.squeeze(torch.stack(x_spikes, dim=1))
		return x_spikes


class LIFEncoder(SpikesEncoder):
	def __init__(
			self,
			n_steps: int,
			n_units: int,
			dt: float = 1e-3,
			device: Optional[torch.device] = None,
			spikes_layer_kwargs: Optional[dict] = None,
	):
		super().__init__(
			n_steps, n_units, LIFLayer, dt, device, spikes_layer_kwargs,
		)
		

class SpyLIFEncoder(SpikesEncoder):
	def __init__(
			self,
			n_steps: int,
			n_units: int,
			dt: float = 1e-3,
			device: Optional[torch.device] = None,
			spikes_layer_kwargs: Optional[dict] = None,
	):
		super().__init__(
			n_steps, n_units, SpyLIFLayer, dt, device, spikes_layer_kwargs,
		)


class ALIFEncoder(SpikesEncoder):
	def __init__(
			self,
			n_steps: int,
			n_units: int,
			dt: float = 1e-3,
			device: Optional[torch.device] = None,
			spikes_layer_kwargs: Optional[dict] = None,
	):
		super().__init__(
			n_steps, n_units, ALIFLayer, dt, device, spikes_layer_kwargs,
		)


