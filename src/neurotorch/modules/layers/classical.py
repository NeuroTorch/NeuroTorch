from typing import Optional, Tuple, Union

import torch
from torch import nn

from .base import BaseNeuronsLayer
from ...dimension import SizeTypes
from ...transforms import to_tensor


class Linear(BaseNeuronsLayer):
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			device: Optional[torch.device] = None,
			**kwargs
	):
		super().__init__(
			input_size=input_size,
			output_size=output_size,
			name=name,
			use_recurrent_connection=False,
			device=device,
			**kwargs
		)
		self.bias_weights = None
		self.activation = self._init_activation(self.kwargs["activation"])
	
	def _set_default_kwargs(self):
		self.kwargs.setdefault("use_bias", True)
		self.kwargs.setdefault("activation", "identity")
	
	def _init_activation(self, activation: Union[torch.nn.Module, str]):
		"""
		Initialise the activation function.

		:param activation: Activation function.
		:type activation: Union[torch.nn.Module, str]
		"""
		str_to_activation = {
			"identity": torch.nn.Identity(),
			"relu"    : torch.nn.ReLU(),
			"tanh"    : torch.nn.Tanh(),
			"sigmoid" : torch.nn.Sigmoid(),
		}
		if isinstance(activation, str):
			assert activation in str_to_activation.keys(), f"Activation {activation} is not implemented."
			self.activation = str_to_activation[activation]
		else:
			self.activation = activation
		return self.activation
	
	def extra_repr(self):
		return f"{', bias' if self.kwargs['use_bias'] else ''}, activation:{self.activation.__class__.__name__}"
	
	def build(self) -> 'Linear':
		if self.kwargs["use_bias"]:
			self.bias_weights = nn.Parameter(
				torch.empty((int(self.output_size),), device=self._device),
				requires_grad=self.requires_grad,
			)
		else:
			self.bias_weights = torch.zeros((int(self.output_size),), dtype=torch.float32, device=self._device)
		super().build()
		self.initialize_weights_()
		return self
	
	def initialize_weights_(self):
		super().initialize_weights_()
		if "bias_weights" in self.kwargs:
			self.bias_weights.data = to_tensor(self.kwargs["bias_weights"]).to(self.device)
		else:
			torch.nn.init.constant_(self.bias_weights, 0.0)
	
	def create_empty_state(
			self,
			batch_size: int = 1,
			**kwargs
	) -> Tuple[torch.Tensor, ...]:
		kwargs.setdefault("n_hh", 0)
		return super().create_empty_state(batch_size=batch_size, **kwargs)
	
	def forward(
			self,
			inputs: torch.Tensor,
			state: Tuple[torch.Tensor, ...] = None,
			**kwargs
	):
		# assert inputs.ndim == 2
		# batch_size, nb_features = inputs.shape
		return self.activation(torch.matmul(inputs, self.forward_weights) + self.bias_weights)
