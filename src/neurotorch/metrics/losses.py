import torch
import torch.nn as nn


class RMSELoss(torch.nn.Module):
	"""
	Class used to compute the RMSE loss.
	
	:math:`\\text{RMSE}(x, y) = \\sqrt{\\frac{1}{n}\\sum_{i=1}^n (x_i - y_i)^2}`
	
	:Attributes:
		- **criterion** (nn.MSELoss): The MSE loss.
	
	"""
	def __init__(self):
		"""
		Constructor for the RMSELoss class.
		"""
		super(RMSELoss, self).__init__()
		self.criterion = nn.MSELoss()

	def forward(self, x, y):
		"""
		Calculate the RMSE loss.
		
		:param x: The first input.
		:param y: The second input.
		
		:return: The RMSE loss.
		"""
		loss = self.criterion(x, y)
		loss = torch.pow(loss + 1e-8, 0.5)
		return loss


class PVarianceLoss(torch.nn.Module):
	"""
	Class used to compute the P-Variance loss.
	
	:math:`\\text{P-Variance}(x, y) = 1 - \\frac{\\text{MSE}(x, y)}{\\text{Var}(y)}`
	
	:Attributes:
		- **criterion** (nn.MSELoss): The MSE loss.
		- **negative** (bool): Whether to return the negative P-Variance loss.
	
	"""
	def __init__(self, negative: bool = False):
		"""
		Constructor for the PVarianceLoss class.
		
		:param negative: Whether to return the negative P-Variance loss.
		:type negative: bool
		"""
		super(PVarianceLoss, self).__init__()
		self.criterion = nn.MSELoss()
		self.negative = negative

	def forward(self, x, y):
		"""
		Calculate the P-Variance loss.
		
		:param x: The first input.
		:param y: The second input.
		
		:return: The P-Variance loss.
		"""
		mse_loss = self.criterion(x, y)
		loss = 1 - mse_loss / torch.var(y)
		if self.negative:
			loss = -loss
		return loss

