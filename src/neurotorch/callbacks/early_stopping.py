import time
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
			
			
class EarlyStoppingOnTime(BaseCallback):
	"""
	Monitor the training process and set the stop_training_flag to True when the threshold is met.
	"""
	CURRENT_SECONDS_COUNT_KEY = "current_seconds_count"
	DELTA_SECONDS_KEY = "delta_seconds"
	
	def __init__(
			self,
			*,
			delta_seconds: float = 10.0 * 60.0,
			resume_on_load: bool = True,
			**kwargs
	):
		"""
		Constructor for EarlyStoppingThreshold class.
		
		:param delta_seconds: The number of seconds to wait before stopping the training.
		:type delta_seconds: float
		:param resume_on_load: Whether to resume the time when loading a checkpoint. If False, the time will be reset
			to 0.
		:type resume_on_load: bool
		:param kwargs: The keyword arguments to pass to the BaseCallback.
		"""
		super().__init__(**kwargs)
		self.delta_seconds = delta_seconds
		self.resume_on_load = resume_on_load
		self.start_time = None
		self.current_seconds_count = 0.0
		
	def load_checkpoint_state(self, trainer, checkpoint: dict, **kwargs):
		self.start_time = None
		self.current_seconds_count = 0.0
		if self.load_state:
			state = checkpoint.get(self.name, {})
			self.delta_seconds = state.get(self.DELTA_SECONDS_KEY, self.delta_seconds)
			if self.resume_on_load:
				self.current_seconds_count = state.get(self.CURRENT_SECONDS_COUNT_KEY, 0.0)
	
	def get_checkpoint_state(self, trainer, **kwargs) -> object:
		if self.save_state:
			state = {
				self.CURRENT_SECONDS_COUNT_KEY: self.current_seconds_count,
				self.DELTA_SECONDS_KEY: self.delta_seconds
			}
			return state
		return None
	
	def start(self, trainer, **kwargs):
		self.start_time = time.time()
		self.update_flags(trainer, **kwargs)
	
	def on_iteration_end(self, trainer, **kwargs):
		self.current_seconds_count += time.time() - self.start_time
		self.update_flags(trainer, **kwargs)
			
	def update_flags(self, trainer, **kwargs):
		if self.current_seconds_count > self.delta_seconds:
			trainer.update_state_(stop_training_flag=True)
