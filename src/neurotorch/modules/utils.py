from typing import Optional, Tuple

import torch

from neurotorch.dimension import SizeTypes
from neurotorch.modules import BaseLayer


class DimensionsCat:
	def __init__(self, start_axis: int, end_axis: int = -1):
		self.start_axis = start_axis
		self.end_axis = end_axis
		assert self.start_axis < self.end_axis, \
			f"Start axis ({self.start_axis}) must be lower than end axis ({self.end_axis})."
		assert all(axis > 0 for axis in self.axes), \
			"Each axis must be greater than 0. The dimension 0 is the batch dimension and can't be concatenate " \
			"with others."

	@property
	def axes(self):
		return [self.start_axis, self.end_axis]

	def __call__(self, inputs: torch.Tensor):
		assert all(axis < inputs.shape[-1] for axis in self.axes), \
			f"Axes ({self.axes}) are not dimension of inputs ({inputs.shape})."
		return torch.flatten(inputs, start_dim=self.start_axis, end_dim=self.end_axis)


class ReduceSum(BaseLayer):
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			axis: int = -1,
			device: Optional[torch.device] = None,
	):
		super().__init__(
			input_size=input_size,
			output_size=output_size,
			name=name,
			device=device,
		)
		self.axis = axis

	def create_empty_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
		pass

	def forward(self, inputs: torch.Tensor, state: torch.Tensor = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
		return inputs.sum(dim=self.axis), None


class ReduceMean(BaseLayer):
	def __init__(self, axis: int = -1):
		super().__init__()
		self.axis = axis

	def create_empty_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
		pass

	def forward(self, inputs: torch.Tensor, state: torch.Tensor = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
		return inputs.mean(dim=self.axis), None








