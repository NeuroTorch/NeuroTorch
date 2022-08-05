import torch
import torch.nn as nn
from typing import Optional, Union, Iterable, Dict
from . import BaseRegularization

# TODO : Unit test -> test_value_t_0

class DaleLawL2(BaseRegularization):
	"""
	Regularisation of the connectome to apply Dale's law. In a nutshell, the Dale's law
	stipulate that neurons can either have excitatory or inhibitory connections, not both.
	This regularisation can reduce the energy of the network and/or allow you to follow the
	Dale's law.
	"""
	def __init__(
			self,
			params: Union[Iterable[torch.nn.Parameter], Dict[str, torch.nn.Parameter]],
			alpha: float = 0.8,
			reference_weights: Optional[torch.Tensor] = None,
			Lambda: float = 1.0,
	):
		"""
		:param alpha: Number between 0 and 1 that favors one of the constraints.
			If alpha = 0 -> Only Dale's law is applied.
			If alpha = 1 -> Only the reduction of the energy is applied.
			If alpha < t < 0 -> Both Dale's law and the reduction of the energy are applied with their ratio.
		:param reference_weights: Reference weights to compare. Must be the same size as the weights. Optional if
		t = 1.
		"""
		super(DaleLawL2, self).__init__(params, Lambda)
		self.alpha = alpha
		if self.alpha > 1 or self.alpha < 0:
			raise ValueError("t must be between 0 and 1")
		if self.alpha != 1 and reference_weights is None:
			raise ValueError("reference_weights must be provided if t != 1")
		self.reference_weights = reference_weights
		if self.reference_weights is not None:
			self.reference_weights = torch.sign(reference_weights)
		else:
			self.reference_weights = torch.tensor(0.0, dtype=torch.float32, device=self.params[0].device)

	def forward(self, *args, **kwargs) -> torch.Tensor:
		"""
		Compute the forward pass of the Dale's law. If alpha = 1 and the reference weights is not provided, it will be
		modified to 0, so it can get cancel.
		:param args: weights matrix
		:param kwargs: kwargs of the forward pass
		"""
		loss_list = []
		for param in self.params:
			loss = torch.trace(param.T @ (self.alpha * param - (1 - self.alpha) * self.reference_weights))
			loss_list.append(loss)
		loss = torch.sum(torch.stack(loss_list))
		return loss
