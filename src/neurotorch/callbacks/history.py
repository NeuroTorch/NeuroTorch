from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from ..callbacks.base_callback import BaseCallback


class TrainingHistory(BaseCallback):
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

	@staticmethod
	def _set_default_plot_kwargs(kwargs: dict):
		kwargs.setdefault('fontsize', 18)
		kwargs.setdefault('linewidth', 4)
		kwargs.setdefault('figsize', (16, 12))
		kwargs.setdefault('dpi', 300)
		return kwargs

	def create_plot(self, **kwargs) -> Tuple[plt.Figure, Dict[str, plt.Axes], Dict[str, plt.Line2D]]:
		kwargs = self._set_default_plot_kwargs(kwargs)
		loss_metrics = [k for k in self._container if 'loss' in k.lower()]
		other_metrics = [k for k in self._container if k not in loss_metrics]
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







