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
	):
		super().__init__()
		self.threshold = threshold
		self.metric = metric
		self.minimize_metric = minimize_metric
	
	def on_iteration_end(self, trainer):
		if self.minimize_metric:
			threshold_met = trainer.training_history[self.metric][-1] < self.threshold
		else:
			threshold_met = trainer.training_history[self.metric][-1] > self.threshold
		if threshold_met:
			trainer.current_training_state = trainer.current_training_state.update(stop_training_flag=True)
