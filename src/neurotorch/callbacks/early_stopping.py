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
	"""
	Monitor the training process and set the stop_training_flag to True when the threshold is met.
	"""
	def __init__(
			self,
			*,
			metric: str,
			threshold: float,
			minimize_metric: bool,
			**kwargs
	):
		"""
		Constructor for EarlyStoppingThreshold class.
		
		:param metric: Name of the metric to monitor.
		:type metric: str
		:param threshold: Threshold value for the metric.
		:type threshold: float
		:param minimize_metric: Whether to minimize or maximize the metric.
		:type minimize_metric: bool
		:param kwargs: The keyword arguments to pass to the BaseCallback.
		"""
		super().__init__(**kwargs)
		self.threshold = threshold
		self.metric = metric
		self.minimize_metric = minimize_metric
	
	def on_iteration_end(self, trainer, **kwargs):
		if self.minimize_metric:
			threshold_met = trainer.current_training_state.itr_metrics[self.metric] < self.threshold
		else:
			threshold_met = trainer.current_training_state.itr_metrics[self.metric] > self.threshold
		if threshold_met:
			trainer.update_state_(stop_training_flag=True)
