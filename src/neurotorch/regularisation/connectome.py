import torch
import torch.nn as nn
from typing import Optional


class DaleLaw(nn.Module):
	"""
	Regularisation of the connectome to apply Dale's law. In a nutshell, the Dale's law
	stipulate that neurons can ether have excitatory or inhibitory connections, not both.
	This regularisation can reduce the energy of the network and/or allow you to follow the
	Dale's law.
	"""
	def __init__(
			self,
			t: float = 0.5,
			reference_weights: Optional[torch.Tensor] = None
	):
		"""
		:param t: Number between 0 and 1 that favors one of the constraints.
			If t = 0 -> Only Dale's law is applied.
			If t = 1 -> Only the reduction of the energy is applied.
			If 1 < t < 0 -> Both Dale's law and the reduction of the energy are applied with their ratio.
		:param reference_weights: Reference weights to compare. Must be the same size as the weights. Optional if
		t = 1.
		"""
		super(DaleLaw, self).__init__()
		self.t = t
		if self.t > 1 or self.t < 0:
			raise ValueError("t must be between 0 and 1")
		if self.t != 1 and reference_weights is None:
			raise ValueError("reference_weights must be provided if t != 1")
		self.reference_weights = reference_weights
		if self.reference_weights is not None:
			self.reference_weights = torch.sign(reference_weights)

	def forward(self, weights: torch.Tensor) -> torch.Tensor:
		"""
		Compute the forward pass of the Dale's law. If t = 1 and the reference weights is not provided, it will be
		modify to 0 so it can get cancel.
		:param weights: weights matrix
		"""
		if self.reference_weights is None:
			self.reference_weights = torch.tensor(0)
		loss = torch.trace(weights.T @ (self.t * weights - (1 - self.t) * self.reference_weights))
		return loss


class DaleLawSimplistic(nn.Module):
	"""
	Regularisation of the connectome to apply Dale's law. In a nutshell, the Dale's law
	stipulate that neurons can ether have excitatory or inhibitory connections, not both.
	This version is a simplistic version of the Dale's law.
	"""
	def __init__(self):
		super(DaleLawSimplistic, self).__init__()

	def forward(self, weights: torch.Tensor) -> torch.Tensor:
		"""
		Apply Dale's law to the connectome. If Dale's law is perfectly applied, the
		loss is 0. If Dale's law is not applied, the loss is N**2
		:param weights: The connectome matrix.
		"""
		N_square = weights.shape[0] * weights.shape[1]
		non_zero_element = torch.count_nonzero(weights)
		zero_element = N_square - non_zero_element
		weights = torch.sign(weights)
		weights = torch.sum(weights, dim=0)
		weights = torch.abs(weights)
		weights = torch.sum(weights) + zero_element
		loss = (-weights + N_square) / (N_square)
		return loss