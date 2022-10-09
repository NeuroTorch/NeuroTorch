import torch
import torch.nn as nn

from ..transforms.base import to_tensor


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
		- :attr:`criterion` (nn.MSELoss): The MSE loss.
		- :attr:`negative` (bool): Whether to return the negative P-Variance loss.
		- :attr:`reduction` (str): The reduction method to use. If 'mean', the output will be averaged. If 'feature', the
			output will be the shape of the last dimension of the input. If 'none', the output will be the same shape as
			the input.
	
	"""
	def __init__(self, negative: bool = False, reduction: str = 'mean', **kwargs):
		"""
		Constructor for the PVarianceLoss class.
		
		:param negative: Whether to return the negative P-Variance loss.
		:type negative: bool
		:param reduction: The reduction method to use. If 'mean', the output will be averaged. If 'feature', the output
			will be the shape of the last dimension of the input. If 'none', the output will be the same shape as the
			input. Defaults to 'mean'.
		:type reduction: str
		:keyword arguments : epsilon: The epsilon value to use to prevent division by zero. Defaults to 1e-5.
		"""
		super(PVarianceLoss, self).__init__()
		assert reduction in ['mean', 'feature', 'none'], 'Reduction must be one of "mean", "feature", or "none".'
		self.reduction = reduction
		mse_reduction = 'mean' if reduction == 'mean' else 'none'
		self.criterion = nn.MSELoss(
			reduction=mse_reduction
		)
		self.negative = negative
		self.epsilon = kwargs.get("epsilon", 1e-5)

	def forward(self, x, y):
		"""
		Calculate the P-Variance loss.
		
		:param x: The first input.
		:param y: The second input.
		
		:return: The P-Variance loss.
		"""
		x, y = to_tensor(x), to_tensor(y)
		if self.reduction == 'feature':
			x_reshape, y_reshape = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
		else:
			x_reshape, y_reshape = x, y
		mse_loss = self.criterion(x_reshape, y_reshape)
		if self.reduction == 'feature':
			mse_loss = mse_loss.mean(dim=0)
			var = y_reshape.var(dim=0)
		else:
			var = y_reshape.var()
		loss = 1 - mse_loss / (var + self.epsilon)
		if self.negative:
			loss = -loss
		return loss
	
	def mean_std_over_batch(self, x, y):
		"""
		Calculate the mean and standard deviation of the P-Variance loss over the batch.
		
		:param x: The first input.
		:param y: The second input.
		
		:return: The mean and standard deviation of the P-Variance loss over the batch.
		"""
		x, y = to_tensor(x), to_tensor(y)
		x_reshape, y_reshape = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
		mse_loss = torch.mean((x_reshape - y_reshape)**2, dim=-1)
		var = y_reshape.var(dim=-1)
		loss = 1 - mse_loss / (var + self.epsilon)
		if self.negative:
			loss = -loss
		return loss.mean(), loss.std()
