from typing import Dict, Iterable, Optional, Union

import torch


class BaseRegularization(torch.nn.Module):
	"""
	Base class for regularization.

	:Attributes:
		(torch.nn.ParameterList) params: The parameters which are regularized.

	"""
	def __init__(
			self,
			params: Union[Iterable[torch.nn.Parameter], Dict[str, torch.nn.Parameter]],
			Lambda: float = 1.0
	):
		"""
		Constructor of the BaseRegularization class.
		:param params: The parameters which are regularized.
		:param Lambda: The weight of the regularization. In other words, the coefficient that multiplies the loss.
		"""
		super(BaseRegularization, self).__init__()
		if isinstance(params, dict):
			self.params = list(params.values())
		else:
			self.params = list(params)
		self.params = torch.nn.ParameterList(self.params)
		self.Lambda = Lambda

	def __call__(self, *args, **kwargs):
		out = super(BaseRegularization, self).__call__(*args, **kwargs)
		return self.Lambda * out

	def forward(self, *args, **kwargs) -> torch.Tensor:
		"""
		Compute the forward pass of the regularization.
		:param args: args of the forward pass
		:param kwargs: kwargs of the forward pass
		"""
		raise NotImplementedError("forward method must be implemented")


class RegularizationList(BaseRegularization):
	"""
	Regularization that applies a list of regularization.
	"""
	def __init__(
			self,
			regularizations: Optional[Iterable[BaseRegularization]] = None,
	):
		"""
		Constructor of the RegularizationList class.
		:param regularizations: The regularizations to apply.
		"""
		self.regularizations = regularizations if regularizations is not None else []
		_params = []
		for regularization in self.regularizations:
			_params.extend(regularization.params)
		super(RegularizationList, self).__init__(
			params=_params,
			Lambda=1.0
		)
		self.regularizations = regularizations if regularizations is not None else []

	def __iter__(self):
		return iter(self.regularizations)

	def forward(self, *args, **kwargs) -> torch.Tensor:
		"""
		Compute the forward pass of the regularization.
		:param args: args of the forward pass
		:param kwargs: kwargs of the forward pass
		"""
		if len(self.regularizations) == 0:
			return torch.tensor(0)
		loss = sum([regularization(*args, **kwargs) for regularization in self.regularizations])
		return loss


class Lp(BaseRegularization):
	"""
	Regularization that applies LK norm.
	"""
	def __init__(
			self,
			params: Union[Iterable[torch.nn.Parameter], Dict[str, torch.nn.Parameter]],
			Lambda: float = 1.0,
			p: int = 1,
	):
		"""
		Constructor of the L1 class.
		:param params: The parameters which are regularized.
		:param p: The k parameter of the LK norm. Example: k=1 -> L1 norm, k=2 -> L2 norm.
		"""
		super(Lp, self).__init__(params, Lambda)
		self.p = p

	def forward(self, *args, **kwargs) -> torch.Tensor:
		"""
		Compute the forward pass of the regularization.
		:param args: args of the forward pass
		:param kwargs: kwargs of the forward pass
		"""
		loss = torch.tensor(0.0, requires_grad=True).to(self.params[0].device)
		for param in self.params:
			loss += torch.linalg.norm(param, self.p)
		return loss


class L1(Lp):
	"""
	Regularization that applies L1 norm.
	"""
	def __init__(
			self,
			params: Union[Iterable[torch.nn.Parameter], Dict[str, torch.nn.Parameter]],
			Lambda: float = 1.0,
	):
		"""
		Constructor of the L1 class.
		:param params: The parameters which are regularized.
		"""
		super(L1, self).__init__(params, Lambda, p=1)


class L2(Lp):
	"""
	Regularization that applies L2 norm.
	"""
	def __init__(
			self,
			params: Union[Iterable[torch.nn.Parameter], Dict[str, torch.nn.Parameter]],
			Lambda: float = 1.0,
	):
		"""
		Constructor of the L2 class.
		:param params: The parameters which are regularized.
		"""
		super(L2, self).__init__(params, Lambda, p=2)


