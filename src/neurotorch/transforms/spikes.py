from typing import Any, Type, Union, Optional

import numpy as np
import torch

from . import ConstantValuesTransform, to_tensor
from .. import LIFLayer, SpyLIFLayer, ALIFLayer, LearningType


class SpikeEncoder(torch.nn.Module):
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
		self.spikes_layer = self.spikes_layer_type(
			n_units, n_units,
			use_recurrent_connection=False,
			learning_type=LearningType.NONE,
			dt=dt,
			device=device,
			**spikes_layer_kwargs,
		)

	def forward(self, x: Any):
		x = to_tensor(x).to(self.spikes_layer.device)
		x = self.const_transform(x)[np.newaxis, :]
		x_spikes = [x[:, 0]]
		x_forward, hh = x[:, 0], None
		for t in range(1, self.n_steps):
			x_forward, hh = self.spikes_layer(x_forward, hh)
			x_spikes.append(x_forward)
		x_spikes = torch.squeeze(torch.stack(x_spikes, dim=1))
		return x_spikes






