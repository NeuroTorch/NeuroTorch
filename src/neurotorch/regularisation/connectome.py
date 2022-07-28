import torch
import torch.nn as nn


class DaleLaw(nn.Module):

	def __init__(self):
		super(DaleLaw, self).__init__()

	def forward(self, weights: torch.Tensor) -> torch.Tensor:
		pass


class DaleLawSimplistic(nn.Module):
	"""
	Regularisation of the connectome to apply Dale's law. In a nutshell, the Dale's law
	stipulate that neurons can ether have excitatory or inhibitory connections, not both.
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