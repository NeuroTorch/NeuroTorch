from typing import Dict, Iterable, Optional, Union
import pythonbasictools as pybt
import torch


class BaseRegularization(torch.nn.Module):
	"""
	Base class for regularization.

	:Attributes:
		- :attr:`params` (torch.nn.ParameterList): The parameters which are regularized.
		- :attr:`Lambda` (float): The weight of the regularization. In other words, the coefficient that multiplies the loss.

	"""
	def __init__(
			self,
			params: Union[Iterable[torch.nn.Parameter], Dict[str, torch.nn.Parameter]],
			Lambda: float = 1.0
	):
		"""
		Constructor of the BaseRegularization class.
		
		:param params: The parameters which are regularized.
		:type params: Union[Iterable[torch.nn.Parameter], Dict[str, torch.nn.Parameter]]
		:param Lambda: The weight of the regularization. In other words, the coefficient that multiplies the loss.
		:type Lambda: float
		"""
		super(BaseRegularization, self).__init__()
		if isinstance(params, dict):
			self.params = list(params.values())
		else:
			self.params = list(params)
		self.params = torch.nn.ParameterList(self.params)
		self.Lambda = Lambda
		self.name = self.__class__.__name__

	def __call__(self, *args, **kwargs) -> torch.Tensor:
		"""
		Call the forward pass of the regularization and scale it by the :attr:`Lambda` attribute.
		
		:param args: args of the forward pass.
		:param kwargs: kwargs of the forward pass.
		
		:return: The loss of the regularization.
		:rtype: torch.Tensor
		"""
		out = super(BaseRegularization, self).__call__(*args, **kwargs)
		return self.Lambda * out

	def forward(self, *args, **kwargs) -> torch.Tensor:
		"""
		Compute the forward pass of the regularization.
		
		:param args: args of the forward pass.
		:param kwargs: kwargs of the forward pass.
		
		:return: The loss of the regularization.
		:rtype: torch.Tensor
		"""
		raise NotImplementedError("forward method must be implemented")


class RegularizationList(BaseRegularization):
	"""
	Regularization that applies a list of regularization.
	
	:Attributes:
		- :attr:`regularizations` (Iterable[BaseRegularization]): The regularizations to apply.
	"""
	def __init__(
			self,
			regularizations: Optional[Iterable[BaseRegularization]] = None,
	):
		"""
		Constructor of the RegularizationList class.
		
		:param regularizations: The regularizations to apply.
		:type regularizations: Optional[Iterable[BaseRegularization]]
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
		"""
		Iterate over the regularizations.
		
		:return: An iterator over the regularizations.
		"""
		return iter(self.regularizations)

	def forward(self, *args, **kwargs) -> torch.Tensor:
		"""
		Compute the forward pass of the regularization.
		
		:param args: args of the forward pass.
		:param kwargs: kwargs of the forward pass.
		"""
		if len(self.regularizations) == 0:
			return torch.tensor(0)
		loss = sum([regularization(*args, **kwargs) for regularization in self.regularizations])
		return loss


# @pybt.docstring.inherit_fields_docstring(fields=["Attributes"], bases=[BaseRegularization])
class Lp(BaseRegularization):
	"""
	Regularization that applies LP norm.
	
	:Attributes:
		- :attr:`p` (int): The p parameter of the LP norm. Example: p=1 -> L1 norm, p=2 -> L2 norm.
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
		:type params: Union[Iterable[torch.nn.Parameter], Dict[str, torch.nn.Parameter]]
		:param Lambda: The weight of the regularization. In other words, the coefficient that multiplies the loss.
		:type Lambda: float
		:param p: The p parameter of the LP norm. Example: p=1 -> L1 norm, p=2 -> L2 norm.
		:type p: int
		"""
		super(Lp, self).__init__(params, Lambda)
		self.p = p

	def forward(self, *args, **kwargs) -> torch.Tensor:
		"""
		Compute the forward pass of the regularization.
		
		:param args: args of the forward pass
		:param kwargs: kwargs of the forward pass
		
		:return: The loss of the regularization.
		:rtype: torch.Tensor
		"""
		loss = torch.tensor(0.0, requires_grad=True).to(self.params[0].device)
		for param in self.params:
			loss += torch.linalg.norm(param, self.p)
		return loss


# @pybt.docstring.inherit_fields_docstring(fields=["Attributes"], bases=[Lp])
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
		:type params: Union[Iterable[torch.nn.Parameter], Dict[str, torch.nn.Parameter]]
		:param Lambda: The weight of the regularization. In other words, the coefficient that multiplies the loss.
		:type Lambda: float
		"""
		super(L1, self).__init__(params, Lambda, p=1)


# @pybt.docstring.inherit_fields_docstring(fields=["Attributes"], bases=[Lp])
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
		:type params: Union[Iterable[torch.nn.Parameter], Dict[str, torch.nn.Parameter]]
		:param Lambda: The weight of the regularization. In other words, the coefficient that multiplies the loss.
		:type Lambda: float
		"""
		super(L2, self).__init__(params, Lambda, p=2)


