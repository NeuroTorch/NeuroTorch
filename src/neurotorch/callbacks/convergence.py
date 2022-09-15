import time

import numpy as np

from .base_callback import BaseCallback


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
		self.threshold_met = False
		self.time_convergence = np.inf
		self.itr_convergence = np.inf
		self.training_time = np.inf
		self.start_time = None
	
	def start(self, trainer):
		self.start_time = time.time()
	
	def close(self, trainer):
		self.training_time = time.time() - self.start_time
	
	def on_iteration_end(self, trainer):
		if not self.threshold_met:
			if self.minimize_metric:
				self.threshold_met = trainer.current_training_state.itr_metrics[self.metric] < self.threshold
			else:
				self.threshold_met = trainer.current_training_state.itr_metrics[self.metric] > self.threshold
			if self.threshold_met:
				self.time_convergence = time.time() - self.start_time
				self.itr_convergence = trainer.current_training_state.iteration
	
	def __repr__(self):
		repr_str = f"ConvergenceTimeGetter("
		repr_str += f"metric={self.metric}, "
		repr_str += f"threshold={self.threshold}, "
		repr_str += f"minimize_metric={self.minimize_metric})"
		repr_str += f"<time_convergence={self.time_convergence} [s], "
		repr_str += f"itr_convergence={self.itr_convergence}, "
		repr_str += f"training_time={self.training_time} [s]>"
		return repr_str



