from typing import Optional, Sequence

import torch

from .learning_algorithm import LearningAlgorithm


class BPTT(LearningAlgorithm):
	r"""
	Apply the backpropagation through time algorithm to the given model.
	"""
	
	def __init__(
			self,
			*,
			params: Optional[Sequence[torch.nn.Parameter]] = None,
			optimizer: Optional[torch.optim.Optimizer] = None,
			**kwargs
	):
		"""
		Constructor for BPTT class.
		
		:param params: The parameters to optimize. If None, the parameters of the model's trainer will be used.
		:type params: Optional[Sequence[torch.nn.Parameter]]
		:param optimizer: The optimizer to use. If not provided, torch.optim.Adam is used.
		:type optimizer: Optional[torch.optim.Optimizer]
		:param kwargs: The keyword arguments to pass to the BaseCallback.
		"""
		super().__init__(**kwargs)
		self.params = params
		self.optimizer = optimizer
		
	def start(self, trainer):
		super().start(trainer)
		if self.params is not None and self.optimizer is None:
			self.optimizer = torch.optim.Adam(self.params)
		elif self.params is None and self.optimizer is not None:
			self.params = [
				param
				for i in range(len(self.optimizer.param_groups))
				for param in self.optimizer.param_groups[i]["params"]
			]
		else:
			self.params = trainer.model.parameters()
			self.optimizer = torch.optim.Adam(self.params)
			
	def on_optimization_begin(self, trainer):
		self.optimizer.zero_grad()
		batch_loss = trainer.current_training_state.batch_loss
		batch_loss.backward()
		self.optimizer.step()
	
	def on_optimization_end(self, trainer):
		self.optimizer.zero_grad()

