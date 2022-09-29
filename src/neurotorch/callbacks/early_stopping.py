from typing import Optional

import numpy as np

from .base_callback import BaseCallback


class EarlyStopping(BaseCallback):
	def __init__(
			self,
			patience: int = 5,
			tol: float = 1e-2,
	):
		self.patience = patience
		self.tol = tol
		self.best_loss = None
		self.counter = 0
	
	def _check_early_stopping(self, patience: int, tol: float = 1e-2) -> bool:
		"""
		:param patience:
		:return:
		"""
		losses = self.loss_history['val'][-patience:]
		return np.all(np.abs(np.diff(losses)) < tol)
		

class EarlyStoppingThreshold(BaseCallback):
	def __init__(
			self,
			*,
			metric: str,
			threshold: float,
			minimize_metric: bool,
			priority: Optional[int] = None,
	):
		super().__init__(priority=priority)
		self.threshold = threshold
		self.metric = metric
		self.minimize_metric = minimize_metric
	
	def on_iteration_end(self, trainer):
		if self.minimize_metric:
			threshold_met = trainer.current_training_state.itr_metrics[self.metric] < self.threshold
		else:
			threshold_met = trainer.current_training_state.itr_metrics[self.metric] > self.threshold
		if threshold_met:
			trainer.update_state_(stop_training_flag=True)
