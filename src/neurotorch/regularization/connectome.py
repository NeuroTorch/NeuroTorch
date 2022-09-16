import torch
import torch.nn as nn
from typing import Optional, Union, Iterable, Dict
import pythonbasictools as pybt
from . import BaseRegularization
from ..init import dale_


# @pybt.docstring.inherit_fields_docstring(fields=["Attributes"], bases=[BaseRegularization])
class DaleLawL2(BaseRegularization):
	"""
	Regularisation of the connectome to apply Dale's law and L2. In a nutshell, the Dale's law
	stipulate that neurons can either have excitatory or inhibitory connections, not both.
	The L2 regularisation reduce the energy of the network. This regularisation allow you to follow the
	Dale's law and/or L2 depending on the factor alpha. The equation used is showed by :eq:`dale_l2`.
	
	.. math::
		:label: dale_l2
		
		\\begin{equation}
			\\mathcal{L}_{\\text{DaleLawL2}} = \\text{Tr}\\left( W^T \\left(\\alpha W - \\left(1 - \\alpha\\right) W_{\\text{ref}}\\right) \\right)
		\\end{equation}
	
	
	In the case where :math:`\\alpha = 0`, the regularisation will only follow the Dale's law shown by :eq:`dale`.
	
	.. math::
		:label: dale
		
		\\begin{equation}
			\\mathcal{L}_{\\text{DaleLaw}} = -\\text{Tr}\\left( W^T W_{\\text{ref}}\\right)
		\\end{equation}
	
	In the case where :math:`\\alpha = 1`, the regularisation will only follow the L2 regularisation shown by :eq:`l2`.
	
	.. math::
		:label: l2
		
		\\begin{equation}
			\\mathcal{L}_{\\text{L2}} = \\text{Tr}\\left( W^T W\\right)
		\\end{equation}
	
	
	:Attributes:
		- :attr:`alpha` (float): Number between 0 and 1 that favors one of the constraints.
		- :attr:`dale_kwargs` (dict): kwargs of the Dale's law. See :func:`dale_`.
		- :attr:`reference_weights` (Iterable[torch.Tensor]): Reference weights to compare. Must be the same size as the weights.
	
	"""
	def __init__(
			self,
			params: Union[Iterable[torch.nn.Parameter], Dict[str, torch.nn.Parameter]],
			alpha: float = 0.8,
			reference_weights: Optional[Iterable[torch.Tensor]] = None,
			Lambda: float = 1.0,
			**dale_kwargs
	):
		"""
		:param params: Weights matrix to regularize (can be multiple)
		:type params: Union[Iterable[torch.nn.Parameter], Dict[str, torch.nn.Parameter]]
		:param alpha: Number between 0 and 1 that favors one of the constraints.
			If alpha = 0 -> Only Dale's law is applied.
			If alpha = 1 -> Only the reduction of the energy is applied.
			If 1 < alpha < 0 -> Both Dale's law and the reduction of the energy are applied with their ratio.
		:type alpha: float
		:param reference_weights: Reference weights to compare. Must be the same size as the weights. If not provided,
			the weights will be generated automatically with the dale_kwargs.
		:type reference_weights: Optional[Iterable[torch.Tensor]]
		:param Lambda: The weight of the regularization. In other words, the coefficient that multiplies the loss.
		:type Lambda: float
		:param dale_kwargs: kwargs of the Dale's law.
		:type dale_kwargs: dict
		
		:keyword float inh_ratio: ratio of inhibitory connections. Must be between 0 and 1.
		:keyword float rho: The connectivity ratio. Must be between 0 and 1. If rho = 1, the tensor will be fully connected.
		:keyword bool inh_first: If True, the inhibitory neurons will be in the first half of the tensor. If False,
			the neurons will be shuffled.
		:keyword Optional[int] seed: seed for the random number generator. If None, the seed is not set.
		"""
		super(DaleLawL2, self).__init__(params, Lambda)
		self.__name__ = self.__class__.__name__
		self.alpha = alpha
		if self.alpha > 1 or self.alpha < 0:
			raise ValueError("alpha must be between 0 and 1")
		self.dale_kwargs = dale_kwargs
		self.reference_weights = self._init_reference_weights(reference_weights)
	
	def _init_reference_weights(
			self,
			reference_weights: Optional[Union[Iterable[torch.Tensor], Dict[str, torch.Tensor]]] = None
	):
		"""
		Initialize the reference weights with Dale's law.
		"""
		if reference_weights is None:
			self.reference_weights = []
			for param in self.params:
				self.reference_weights.append(torch.sign(dale_(torch.empty_like(param), **self.dale_kwargs)))
		else:
			self.reference_weights = [torch.sign(ref) for ref in reference_weights]
		return self.reference_weights

	def forward(self, *args, **kwargs) -> torch.Tensor:
		"""
		Compute the forward pass of the Dale's law. If alpha = 1 and the reference weights is not provided, it will be
		modified to 0, so it can get cancel.
		
		:param args: weights matrix
		:param kwargs: kwargs of the forward pass
		"""
		loss_list = []
		for param, ref in zip(self.params, self.reference_weights):
			loss = torch.trace(
				param.T @ (self.alpha * param - (1 - self.alpha) * ref.to(param.device))
			)
			loss_list.append(loss)
		if len(self.params) == 0:
			loss = torch.tensor(0.0, dtype=torch.float32)
		else:
			loss = torch.sum(torch.stack(loss_list))
		return loss


class DaleLaw(DaleLawL2):
	def __init__(
			self,
			params: Union[Iterable[torch.nn.Parameter], Dict[str, torch.nn.Parameter]],
			reference_weights: Optional[Iterable[torch.Tensor]] = None,
			Lambda: float = 1.0,
			**dale_kwargs
	):
		"""
		:param params: Weights matrix to regularize (can be multiple)
		:param reference_weights: Reference weights to compare. Must be the same size as the weights. If not provided,
			the weights will be generated automatically with the dale_kwargs.
		:param Lambda: The weight of the regularization. In other words, the coefficient that multiplies the loss.
		:param dale_kwargs: kwargs of the Dale's law.

		:keyword inh_ratio: ratio of inhibitory connections. Must be between 0 and 1.
		:keyword rho: The connectivity ratio. Must be between 0 and 1. If rho = 1, the tensor will be fully connected.
		:keyword inh_first: If True, the inhibitory neurons will be in the first half of the tensor. If False,
			the neurons will be shuffled.
		:keyword seed: seed for the random number generator. If None, the seed is not set.
		"""
		super(DaleLaw, self).__init__(params, 0.0, reference_weights, Lambda, **dale_kwargs)
		self.__name__ = self.__class__.__name__

