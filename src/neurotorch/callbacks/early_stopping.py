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
		


