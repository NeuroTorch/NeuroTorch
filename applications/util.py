import time
from typing import Type, Optional

import numpy as np
import torch
import neurotorch as nt
from neurotorch.regularization import BaseRegularization
from neurotorch.callbacks.base_callback import BaseCallback


class ConvergenceTimeGetter(BaseCallback):
	def __init__(
			self,
			*,
			metric: str,
			threshold: float,
			minimize_metric: bool,
	):
		super().__init__()
		self.threshold = threshold
		self.metric = metric
		self.minimize_metric = minimize_metric
		self.time_convergence = None
		self.itr_convergence = None
		self.training_time = None
		self.start_time = None
	
	def start(self, trainer):
		self.start_time = time.time()

	def close(self, trainer):
		self.training_time = time.time() - self.start_time
	
	def on_iteration_end(self, trainer):
		if self.time_convergence is None:
			if self.minimize_metric:
				threshold_met = trainer.training_history[self.metric][-1] < self.threshold
			else:
				threshold_met = trainer.training_history[self.metric][-1] > self.threshold
			if threshold_met:
				self.time_convergence = time.time() - self.start_time
				self.itr_convergence = trainer.current_training_state.iteration


def get_optimizer(optimizer_name: str) -> Type[torch.optim.Optimizer]:
	name_to_opt = {
		"sgd": torch.optim.SGD,
		"adam": torch.optim.Adam,
		"adamax": torch.optim.Adamax,
		"rmsprop": torch.optim.RMSprop,
		"adagrad": torch.optim.Adagrad,
		"adadelta": torch.optim.Adadelta,
		"adamw": torch.optim.AdamW,
	}
	return name_to_opt[optimizer_name.lower()]


def get_regularization(
		regularization_name: Optional[str],
		parameters,
		**kwargs
) -> Optional[BaseRegularization]:
	if regularization_name is None or not regularization_name:
		return None
	regs = regularization_name.lower().split('_')
	name_to_reg = {
		"l1": nt.L1,
		"l2": nt.L2,
		"dale": nt.DaleLaw,
	}
	return nt.RegularizationList([name_to_reg[reg](parameters, **kwargs) for reg in regs])

