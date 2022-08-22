from typing import List, Optional, Iterable

import numpy as np

from neurotorch.callbacks.base_callback import BaseCallback


class LinearLRScheduler(BaseCallback):
	def __init__(self, lr_start: float, lr_end: float, n_steps: int):
		super().__init__()
		self.lr_start = lr_start
		self.lr_end = lr_end
		self.n_steps = n_steps
		self.lr = self.lr_start
		self.lr_decay = (self.lr_start - self.lr_end) / self.n_steps
	
	def on_iteration_end(self, trainer):
		trainer.training_history.append('lr', self.lr)
		step = trainer.current_training_state.iteration
		self.lr = max(self.lr_start - step * self.lr_decay, self.lr_end)
		for g in trainer.optimizer.param_groups:
			g['lr'] = self.lr


class LRSchedulerOnMetric(BaseCallback):
	"""
	Class to schedule the learning rate of the optimizer based on the metric value.
	Each time the metric reach the next value of the schedule, the learning rate is multiplied by the given decay.
	The learning rate is also capped at the given minimum value.
	"""
	def __init__(
			self,
			metric: str,
			metric_schedule: Iterable[float],
			*,
			minimize_metric: Optional[bool] = None,
			lr_decay: Optional[float] = None,
			min_lr: float = 1e-12,
			lr_start: float = None,
	):
		"""
		Initialize the scheduler with the given metric and metric schedule.
		
		:param metric: The metric to use to schedule the learning rate.
		:param metric_schedule: The schedule of the metric.
		:param minimize_metric: Whether to minimize the metric or maximize it. If None, infer from the metric schedule.
		:param lr_decay: The decay factor to use when the metric reach the next value of the schedule. If None, the
		decay is computed automatically as (lr_start - min_lr) / len(metric_schedule).
		:param min_lr: The minimum learning rate to use.
		:param lr_start: The learning rate to use at the beginning of the training. If None, the learning rate is
		get automatically as the learning rate of the first group of the optimizer.
		"""
		super().__init__()
		self.metric = metric
		self.metric_schedule = np.asarray(metric_schedule)
		self._check_schedule_ascending_or_descending()
		self.minimize_metric = minimize_metric
		self._init_minimize_metric()
		self.lr_decay = lr_decay
		self.min_lr = min_lr
		self.lr_start = lr_start
		self.lr = self.lr_start
	
	def on_iteration_end(self, trainer):
		last_metric = trainer.training_history[self.metric][-1]
		trainer.training_history.append('lr', self.lr)
		self.lr = max(self.lr_start - self.lr_decay * self.get_step(last_metric), self.min_lr)
		for g in trainer.optimizer.param_groups:
			g['lr'] = self.lr
	
	def _check_schedule_ascending_or_descending(self):
		sig_diff = np.sign(np.diff(self.metric_schedule))
		all_positive = np.all(sig_diff >= 0)
		all_negative = np.all(sig_diff <= 0)
		if not (all_positive or all_negative):
			raise ValueError('The metric schedule must be ascending or descending.')
		if all_positive:
			self.metric_schedule = np.concatenate(([-np.inf], self.metric_schedule))
		if all_negative:
			self.metric_schedule = np.concatenate(([np.inf], self.metric_schedule))
	
	def _init_minimize_metric(self):
		if self.minimize_metric is None:
			self.minimize_metric = np.all(np.diff(self.metric_schedule) <= 0)
	
	def _init_lr_decay(self):
		if self.lr_decay is None:
			self.lr_decay = (self.lr_start - self.min_lr) / len(self.metric_schedule)
	
	def get_step(self, last_metric):
		last_index = len(self.metric_schedule) - 1
		if self.minimize_metric:
			return last_index - np.argmax((last_metric <= self.metric_schedule)[::-1])
		else:
			return last_index - np.argmax((last_metric >= self.metric_schedule)[::-1])
	
	def start(self, trainer):
		if self.lr_start is None:
			self.lr_start = trainer.optimizer.param_groups[0]['lr']
		self.lr = self.lr_start
		self._init_lr_decay()
