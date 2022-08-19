from neurotorch.callbacks.base_callback import BaseCallback


class LinearLRScheduler(BaseCallback):
	def __init__(self, lr_start: float, lr_end: float, n_steps: int):
		super().__init__()
		self.lr_start = lr_start
		self.lr_end = lr_end
		self.n_steps = n_steps
		self.step = 0
		self.lr = self.lr_start
		self.lr_decay = (self.lr_start - self.lr_end) / self.n_steps
	
	def on_iteration_end(self, trainer):
		trainer.training_history.append('lr', self.lr)
		self.step += 1
		self.lr = max(self.lr_start - self.step * self.lr_decay, self.lr_end)
		for g in trainer.optimizer.param_groups:
			g['lr'] = self.lr




