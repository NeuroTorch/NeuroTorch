from typing import Any, Type, Union, Optional

import numpy as np
import torch

from . import ConstantValuesTransform, to_tensor
from ..modules.layers import LIFLayer, SpyLIFLayer, ALIFLayer, LearningType


class SpikesEncoder(torch.nn.Module):
	def __init__(
			self,
			n_steps: int,
			n_units: int,
			spikes_layer_type: Type[Union[LIFLayer, SpyLIFLayer, ALIFLayer]],
			dt: float = 1e-3,
			learning_type: LearningType = LearningType.NONE,
			device: Optional[torch.device] = None,
			spikes_layer_kwargs: Optional[dict] = None,
	):
		"""
		Constructor for the SpikesEncoder class.
		
		:param n_steps: Number of steps to encode.
		:param n_units: Number of units in the encoder.
		:param spikes_layer_type: Dynamic of the spikes layer.
		:param dt: Time step of the simulation.
		:param learning_type: Learning type of the spikes layer.
		:param device: Device to use for the encoder.
		:param spikes_layer_kwargs: Keyword arguments for the spikes layer.
		
		:keyword spikes_layer_kwargs:
			* <forward_weights>: Forward weights of the spikes layer. Default: eye(n_units).
			* <use_recurrent_connection>: Whether to use a recurrent connection. Default: False.
			* <name>: Name of the spikes layer. Default: "encoder".
		
		"""
		super().__init__()
		self.n_steps = n_steps
		self.const_transform = ConstantValuesTransform(n_steps, batch_wise=True)
		self.spikes_layer_type = spikes_layer_type
		self.dt = dt
		self.learning_type = learning_type
		if spikes_layer_kwargs is None:
			spikes_layer_kwargs = dict(
				forward_weights=torch.eye(n_units),
				use_recurrent_connection=False,
				name='encoder',
			)
		
		assert "learning_type" not in spikes_layer_kwargs, \
			"learning_type cannot be specified since it must be the given value"
		assert "dt" not in spikes_layer_kwargs, \
			"dt cannot be specified since it must be the given dt"
		assert "device" not in spikes_layer_kwargs, \
			"device cannot be specified since it must be the given device"
		
		self.spikes_layer = self.spikes_layer_type(
			n_units, n_units,
			learning_type=self.learning_type,
			dt=dt,
			device=device,
			**spikes_layer_kwargs,
		)
		self.spikes_layer.build()
	
	def to(self, device):
		self.spikes_layer.device = device
		return super().to(device)

	def forward(self, x: Any):
		x = to_tensor(x).to(self.spikes_layer.device)
		x = self.const_transform(x)
		x_spikes, hh = [], None
		for t in range(self.n_steps):
			spikes, hh = self.spikes_layer(x[:, t], hh)
			x_spikes.append(spikes)
		# x_spikes = torch.squeeze(torch.stack(x_spikes, dim=1))
		return torch.stack(x_spikes, dim=1)


class LIFEncoder(SpikesEncoder):
	def __init__(
			self,
			n_steps: int,
			n_units: int,
			dt: float = 1e-3,
			learning_type: LearningType = LearningType.NONE,
			device: Optional[torch.device] = None,
			spikes_layer_kwargs: Optional[dict] = None,
	):
		super().__init__(
			n_steps, n_units, LIFLayer, dt, learning_type, device, spikes_layer_kwargs,
		)


class SpyLIFEncoder(SpikesEncoder):
	def __init__(
			self,
			n_steps: int,
			n_units: int,
			dt: float = 1e-3,
			learning_type: LearningType = LearningType.NONE,
			device: Optional[torch.device] = None,
			spikes_layer_kwargs: Optional[dict] = None,
	):
		super().__init__(
			n_steps, n_units, SpyLIFLayer, dt, learning_type, device, spikes_layer_kwargs,
		)


class ALIFEncoder(SpikesEncoder):
	def __init__(
			self,
			n_steps: int,
			n_units: int,
			dt: float = 1e-3,
			learning_type: LearningType = LearningType.NONE,
			device: Optional[torch.device] = None,
			spikes_layer_kwargs: Optional[dict] = None,
	):
		super().__init__(
			n_steps, n_units, ALIFLayer, dt, learning_type, device, spikes_layer_kwargs,
		)


