import torch

from .base import to_tensor


class MeanConv(torch.nn.Module):
	def __init__(
			self,
			kernel_size: int,
			alpha: float = 1.0,
			learn_alpha: bool = True,
			learn_kernel: bool = True,
			activation: torch.nn.Module = torch.nn.Identity(),
	):
		super(MeanConv, self).__init__()
		self.kernel_size = kernel_size
		self.learn_kernel = learn_kernel
		self.kernel = torch.nn.Parameter(torch.ones(1, self.kernel_size, 1), requires_grad=learn_kernel)
		self.alpha = alpha
		self.learn_alpha = learn_alpha
		if learn_alpha:
			self.alpha = torch.nn.Parameter(to_tensor(self.alpha))
		self.activation = activation
	
	def forward(self, inputs: torch.Tensor):
		batch_size, time_steps, n_units = inputs.shape
		inputs_view = torch.reshape(inputs, (batch_size, -1, self.kernel_size, n_units))
		inputs_mean = self.alpha * torch.sum(self.kernel * inputs_view, dim=2) / self.kernel_size
		return self.activation(inputs_mean)








