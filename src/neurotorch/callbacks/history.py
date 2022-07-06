from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from ..callbacks.base_callback import BaseCallback


class TrainingHistory(BaseCallback):
	def __init__(self, container: Dict[str, List[float]] = None):
		self.container = defaultdict(list)
		if container is not None:
			self.container.update(container)

	def __getitem__(self, item):
		return self.container[item]

	def __setitem__(self, key, value):
		self.container[key] = value

	def __contains__(self, item):
		return item in self.container

	def __iter__(self):
		return iter(self.container)

	def __len__(self):
		if len(self.container) == 0:
			return 0
		return len(self.container[list(self.container.keys())[0]])

	def items(self):
		return self.container.items()

	def concat(self, other):
		for key, values in other.items():
			if isinstance(values, list):
				self.container[key].extend(values)
			else:
				self.container[key].append(values)

	def append(self, key, value):
		self.container[key].append(value)

	def min(self, key=None):
		if key is None:
			key = list(self.container.keys())[0]
		if key in self:
			return min(self[key])
		return np.inf

	def min_item(self, key=None):
		if key is None:
			key = list(self.container.keys())[0]
		if key in self:
			argmin = np.argmin(self[key])
			return {k: v[argmin] for k, v in self.items()}
		raise ValueError("key not in container")

	def max(self, key=None):
		if key is None:
			key = list(self.container.keys())[0]
		if key in self:
			return max(self[key])
		return -np.inf

	def max_item(self, key=None):
		if key is None:
			key = list(self.container.keys())[0]
		if key in self:
			argmax = np.argmax(self[key])
			return {k: v[argmax] for k, v in self.items()}
		raise ValueError("key not in container")

	@staticmethod
	def _set_default_plot_kwargs(kwargs: dict):
		kwargs.setdefault('fontsize', 18)
		kwargs.setdefault('linewidth', 4)
		kwargs.setdefault('figsize', (16, 12))
		kwargs.setdefault('dpi', 300)
		return kwargs

	def create_plot(self, **kwargs) -> Tuple[plt.Figure, Dict[str, plt.Axes], Dict[str, plt.Line2D]]:
		kwargs = self._set_default_plot_kwargs(kwargs)
		loss_metrics = [k for k in self.container if 'loss' in k.lower()]
		other_metrics = [k for k in self.container if k not in loss_metrics]
		n_cols = int(np.sqrt(1 + len(other_metrics)))
		n_rows = int(np.ceil((1 + len(other_metrics)) / n_cols))
		axes_dict, lines = {}, {}
		fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=kwargs["figsize"], sharex='all')
		axes = np.ravel(axes)
		for i, ax in enumerate(axes):
			if i >= 1 + len(other_metrics):
				ax.axis('off')
				continue
			if i == 0:
				for k in loss_metrics:
					lines[k] = ax.plot(self[k], label=k, linewidth=kwargs['linewidth'])[0]
				axes_dict['losses'] = ax
				ax.set_ylabel("Loss [-]", fontsize=kwargs["fontsize"])
				ax.set_xlabel("Iterations [-]", fontsize=kwargs["fontsize"])
				ax.legend(fontsize=kwargs["fontsize"])
			else:
				k = other_metrics[i - 1]
				lines[k] = ax.plot(self[k], label=k, linewidth=kwargs['linewidth'])[0]
				axes_dict[k] = ax
				ax.set_xlabel("Iterations [-]", fontsize=kwargs["fontsize"])
				ax.legend(fontsize=kwargs["fontsize"])
		return fig, axes_dict, lines

	def plot(
			self,
			save_path=None,
			show=False,
			**kwargs
	) -> Tuple[plt.Figure, Dict[str, plt.Axes], Dict[str, plt.Line2D]]:
		kwargs = self._set_default_plot_kwargs(kwargs)
		fig, axes, lines = self.create_plot(**kwargs)
		if save_path is not None:
			fig.savefig(save_path, dpi=kwargs["dpi"])
		if show:
			plt.show(block=kwargs.get('block', True))
		if kwargs.get('close', False):
			plt.close(fig)
		return fig, axes, lines

	def update_fig(
			self,
			fig: plt.Figure,
			axes: Dict[str, plt.Axes],
			lines: Dict[str, plt.Line2D],
			**kwargs
	) -> Tuple[plt.Figure, Dict[str, plt.Axes], Dict[str, plt.Line2D]]:
		kwargs = self._set_default_plot_kwargs(kwargs)
		for k in lines:
			lines[k].set_data(range(len(self[k])), self[k])
		for k in axes:
			axes[k].relim()
			axes[k].autoscale_view()
		fig.canvas.draw()
		fig.canvas.flush_events()
		return fig, axes, lines

	def on_iteration_end(self, trainer):
		self.concat(trainer.current_training_state.itr_metrics)







