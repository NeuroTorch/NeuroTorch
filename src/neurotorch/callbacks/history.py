from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from ..callbacks.base_callback import BaseCallback


class TrainingHistory(BaseCallback):

	@staticmethod
	def _set_default_plot_kwargs(kwargs: dict):
		kwargs.setdefault('fontsize', 18)
		kwargs.setdefault('linewidth', 4)
		kwargs.setdefault('figsize', (16, 12))
		kwargs.setdefault('dpi', 300)
		return kwargs

	@staticmethod
	def _remove_prefix_from_metrics(metrics: List[str]):
		return [metric.split('_')[-1] for metric in metrics]

	def __init__(self, container: Dict[str, List[float]] = None, default_value=np.NAN):
		"""
		Initialize the container with the given container.
		:param container: The container to initialize the container with.
		:param default_value: The default value to use to equalize the lengths of the container's items.
		"""
		self._container = {}
		self.default_value = default_value
		self._length = 0
		if container is not None:
			self.concat(container)

	def __getitem__(self, key):
		self._add_key(key)
		return self._container[key]

	def __setitem__(self, key, value: list):
		self._add_key(key)
		assert len(value) == len(self), "Length of value must be equal to length of container"
		self._container[key] = list(value)

	def __contains__(self, item):
		return item in self._container

	def __iter__(self):
		return iter(self._container)

	def __len__(self):
		return self._length

	def _add_key(self, key):
		if key not in self._container:
			self._container[key] = [self.default_value] * len(self)

	def _increase_to_size(self, to_size: int):
		"""
		Increase the size of the container items to the given size.
		:param to_size: The size to increase the container to.
		:return: None
		"""
		if len(self) > to_size:
			raise ValueError("Cannot increase size of container to smaller size")
		if len(self) == to_size:
			return
		for key, values in self.items():
			self[key].extend([self.default_value] * (to_size - len(self)))
		self._length = to_size

	def keys(self):
		return self._container.keys()

	def items(self):
		return self._container.items()

	def concat(self, other):
		self.insert(len(self), other)

	def insert(self, index: int, other):
		"""
		Increase the size of the container items to the given index and insert the given other into the container.
		:param index: The index to insert the other at.
		:param other: The other to insert.
		:return: None
		"""
		if index >= len(self):
			self._increase_to_size(index + 1)
		for key, values in other.items():
			if isinstance(values, list):
				raise NotImplementedError()
			else:
				self[key][index] = values

	def append(self, key, value):
		self.insert(len(self), {key: value})

	def min(self, key=None, default=np.inf):
		if key is None:
			key = list(self.keys())[0]
		if key in self:
			return np.nanmin(self[key])
		return default

	def min_item(self, key=None):
		if key is None:
			key = list(self.keys())[0]
		if key in self:
			argmin = np.argmin(self[key])
			return {k: v[argmin] for k, v in self.items()}
		raise ValueError("key not in container")

	def max(self, key=None, default=-np.inf):
		if key is None:
			key = list(self.keys())[0]
		if key in self:
			return np.nanmax(self[key])
		return default

	def max_item(self, key=None):
		if key is None:
			key = list(self.keys())[0]
		if key in self:
			argmax = np.argmax(self[key])
			return {k: v[argmax] for k, v in self.items()}
		raise ValueError("key not in container")

	def create_plot(self, **kwargs) -> Tuple[plt.Figure, Dict[str, plt.Axes], Dict[str, plt.Line2D]]:
		kwargs = self._set_default_plot_kwargs(kwargs)
		keys_lower_to_given = {k.lower(): k for k in self.keys()}
		keys_lower = [key.lower() for key in self.keys()]
		loss_metrics = [k for k in keys_lower if 'loss' in k]
		keys_lower = list(set(keys_lower) - set(loss_metrics))
		val_metrics = [k for k in keys_lower if 'val' in k]
		train_metrics = [k for k in keys_lower if 'train' in k]
		test_metrics = [k for k in keys_lower if 'test' in k]
		n_set_metrics = max(len(val_metrics), len(train_metrics), len(test_metrics))
		max_set_metrics_container = [c for c in [val_metrics, train_metrics, test_metrics] if len(c) == n_set_metrics][0]
		other_metrics = list(set(keys_lower) - set(val_metrics) - set(train_metrics) - set(test_metrics))
		n_graphs = 1 + n_set_metrics + len(other_metrics)
		n_cols = int(np.sqrt(n_graphs))
		n_rows = int(np.ceil(n_graphs / n_cols))
		axes_dict, lines = {}, {}
		fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=kwargs["figsize"], sharex='all')
		axes = np.ravel(axes)
		for i, ax in enumerate(axes):
			if i >= n_graphs:
				ax.axis('off')
				continue
			if i == 0:
				for k in loss_metrics:
					key = keys_lower_to_given[k]
					lines[key] = ax.plot(self[key], label=key, linewidth=kwargs['linewidth'])[0]
				axes_dict['losses'] = ax
				ax.set_ylabel("Loss [-]", fontsize=kwargs["fontsize"])
				ax.set_xlabel("Iterations [-]", fontsize=kwargs["fontsize"])
				ax.legend(fontsize=kwargs["fontsize"])
			elif 0 < i <= n_set_metrics:
				metric_basename = '_'.join(max_set_metrics_container[i-1].split('_')[1:])
				for prefix in ['val', 'train', 'test']:
					k = prefix + '_' + metric_basename
					key = keys_lower_to_given[k]
					if key in self:
						lines[key] = ax.plot(self[key], label=key, linewidth=kwargs['linewidth'])[0]
						axes_dict[key] = ax
				ax.set_xlabel("Iterations [-]", fontsize=kwargs["fontsize"])
				ax.legend(fontsize=kwargs["fontsize"])
			else:
				k = other_metrics[i - 1 - n_set_metrics]
				key = keys_lower_to_given[k]
				lines[key] = ax.plot(self[key], label=key, linewidth=kwargs['linewidth'])[0]
				axes_dict[key] = ax
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
		plt.close('all')
		fig, axes, lines = self.create_plot(**kwargs)
		if save_path is not None:
			fig.savefig(save_path, dpi=kwargs["dpi"])
		if show:
			plt.show(block=kwargs.get('block', True))
		if kwargs.get('close', True):
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
		self.insert(trainer.current_training_state.iteration, trainer.current_training_state.itr_metrics)







