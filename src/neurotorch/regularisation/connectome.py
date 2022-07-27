import torch
import torch.nn as nn


class DaleLaw(nn.Module):
	def __init__(self):
		super(DaleLaw, self).__init__()

	def forward(self, weights: torch.Tensor) -> torch.Tensor:
		# compute_inhibitory_weights = torch.sign(weights)
		# return distance between entre 0.2 and the ratio (L2)
		# normalize between 0 and 1
		return None