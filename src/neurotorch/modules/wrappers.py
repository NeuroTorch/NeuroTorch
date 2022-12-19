from typing import Optional

import torch
from .base import NamedModule


class NamedModuleWrapper(NamedModule):
	"""Base class for all wrappers."""

	def __init__(self, module: torch.nn.Module, name: Optional[str] = None):
		super().__init__(name=name)
		self.module = module
	
	def forward(self, *args, **kwargs):
		return self.module(*args, **kwargs)
