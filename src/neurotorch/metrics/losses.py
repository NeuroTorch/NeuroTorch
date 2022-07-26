import torch
import torch.nn as nn


class RMSELoss(torch.nn.Module):
	def __init__(self):
		super(RMSELoss, self).__init__()
		self.criterion = nn.MSELoss()

	def forward(self, x, y):
		loss = self.criterion(x, y)
		loss = torch.pow(loss + 1e-8, 0.5)
		return loss
