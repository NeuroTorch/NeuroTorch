import os
from collections import defaultdict
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from .curriculum import Curriculum
from ..callbacks import TrainingHistory
from ..utils import legend_without_duplicate_labels_


class TrainingHistoriesMap:
	REPORT_KEY = "report"

	def __init__(self, curriculum: Optional[Curriculum] = None):
		self.curriculum = curriculum
		self.histories = defaultdict(TrainingHistory, **{TrainingHistoriesMap.REPORT_KEY: TrainingHistory()})

	@property
	def report_history(self) -> TrainingHistory:
		return self.histories[TrainingHistoriesMap.REPORT_KEY]

	def max(self, key=None):
		if self.curriculum is None:
			return self.histories[TrainingHistoriesMap.REPORT_KEY].max(key)
		else:
			return self.histories[self.curriculum.current_lesson.name].max(key)

	def concat(self, other):
		self.histories[TrainingHistoriesMap.REPORT_KEY].concat(other)
		if self.curriculum is not None:
			return self.histories[self.curriculum.current_lesson.name].concat(other)

	def append(self, key, value):
		self.histories[TrainingHistoriesMap.REPORT_KEY].append(key, value)
		if self.curriculum is not None:
			return self.histories[self.curriculum.current_lesson.name].append(key, value)

	@staticmethod
	def _set_default_plot_kwargs(kwargs: dict):
		kwargs.setdefault('fontsize', 16)
		kwargs.setdefault('linewidth', 3)
		kwargs.setdefault('figsize', (16, 12))
		kwargs.setdefault('dpi', 300)
		return kwargs

	def plot(self, save_path=None, show=False, lesson_idx: Optional[Union[int, str]] = None, **kwargs):
		kwargs = self._set_default_plot_kwargs(kwargs)
		if self.curriculum is None:
			assert lesson_idx is None, "lesson_idx must be None if curriculum is None"
			return self.plot_history(TrainingHistoriesMap.REPORT_KEY, save_path, show, **kwargs)
		if lesson_idx is None:
			self.plot_history(TrainingHistoriesMap.REPORT_KEY, save_path, show, **kwargs)
		else:
			self.plot_history(self.curriculum[lesson_idx].name, save_path, show, **kwargs)

	def plot_history(
			self,
			history_name: str,
			save_path=None,
			show=False,
			**kwargs
	):
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		history = self.histories[history_name]
		if self.curriculum is not None and history_name != TrainingHistoriesMap.REPORT_KEY:
			lessons = [self.curriculum[history_name]]
			lessons_start_itr = [0]
		elif self.curriculum is not None and history_name == TrainingHistoriesMap.REPORT_KEY:
			lessons = self.curriculum.lessons
			lessons_lengths = {k: [len(self.histories[lesson.name][k]) for lesson in lessons] for k in history._container}
			lessons_start_itr = {k: np.cumsum(lessons_lengths[k]) for k in history.keys()}
		else:
			lessons = []
			lessons_start_itr = []

		kwargs = self._set_default_plot_kwargs(kwargs)
		loss_metrics = [k for k in history.keys() if 'loss' in k.lower()]
		rewards_metrics = [k for k in history.keys() if 'reward' in k.lower()]
		other_metrics = [k for k in history.keys() if k not in loss_metrics and k not in rewards_metrics]
		n_metrics = 2 + len(other_metrics)
		n_cols = int(np.sqrt(n_metrics))
		n_rows = int(n_metrics / n_cols)
		fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=kwargs["figsize"], sharex='all')
		if axes.ndim == 1:
			axes = np.expand_dims(axes, axis=-1)
		for row_i in range(n_rows):
			for col_i in range(n_cols):
				ax = axes[row_i, col_i]
				ravel_index = row_i * n_cols + col_i
				if ravel_index == 0:
					for k in loss_metrics:
						ax.plot(history[k], label=k, linewidth=kwargs['linewidth'])
					ax.set_ylabel("Loss [-]", fontsize=kwargs["fontsize"])
					ax.legend(fontsize=kwargs["fontsize"])
				elif ravel_index == 1:
					for k in rewards_metrics:
						ax.plot(history[k], label=k, linewidth=kwargs['linewidth'])
						for lesson_idx, lesson in enumerate(lessons):
							if lesson.completion_criteria.measure == k:
								ax.plot(
									lesson.completion_criteria.threshold*np.ones(len(history[k])), 'k--',
									label=f"{k} threshold", linewidth=kwargs['linewidth']
								)
							if history_name == TrainingHistoriesMap.REPORT_KEY and lesson.is_completed:
								ax.axvline(
									lessons_start_itr[k][lesson_idx], ymin=np.min(history[k]), ymax=np.max(history[k]),
									color='r', linestyle='--', linewidth=kwargs['linewidth'], label=f"lesson start"
								)
					ax.set_ylabel("Rewards [-]", fontsize=kwargs["fontsize"])
					ax.legend(fontsize=kwargs["fontsize"])
				else:
					k = other_metrics[ravel_index - 1]
					ax.plot(history[k], label=k, linewidth=kwargs['linewidth'])
					ax.legend(fontsize=kwargs["fontsize"])
				if row_i == n_rows - 1:
					ax.set_xlabel("Iterations [-]", fontsize=kwargs["fontsize"])
				legend_without_duplicate_labels_(ax)
		if save_path is not None:
			plt.savefig(save_path, dpi=kwargs["dpi"])
		if show:
			plt.show()
		plt.close(fig)




