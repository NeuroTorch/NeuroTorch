import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from ..transforms.base import to_numpy
from ..callbacks.base_callback import BaseCallback


class TrainingHistory(BaseCallback):
    """
    This class is used to store some metrics over the training process.

    :Attributes:
        - **default_value** (float): The default value to use to equalize the lengths of the container's items.
    """
    DEFAULT_PRIORITY = BaseCallback.DEFAULT_HIGH_PRIORITY

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

    def __init__(
            self,
            container: Dict[str, List[float]] = None,
            default_value=np.NAN,
            **kwargs
    ):
        """
        Initialize the container with the given container.

        :param container: The container to initialize the container with.
        :type container: Dict[str, List[float]]
        :param default_value: The default value to use to equalize the lengths of the container's items.
        :type default_value: float
        :param kwargs: The keyword arguments to pass to the BaseCallback.
        """
        super().__init__(**kwargs)
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
        :type to_size: int

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
        :type index: int
        :param other: The other to insert.
        :type other: Dict[str, float]

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
        """
        Add the given value to the given key. Increase the size of the container by one.

        :param key: The key to add the value to.
        :type key: str
        :param value: The value to add.
        :type value: float

        :return: None
        """
        self.insert(len(self), {key: value})

    def min(self, key=None, default: float = np.inf) -> float:
        """
        Get the minimum value of the given key.

        :param key: The key to get the minimum value of. If None, the first key is used.
        :type key: str
        :param default: The default value to return if the key is not in the container.
        :type default: float

        :return: The minimum value of the given key.
        :rtype: float
        """
        if key is None:
            key = list(self.keys())[0]
        if key in self and len(self[key]) > 0:
            return np.nanmin(self[key])
        return default

    def min_item(self, key=None) -> Dict[str, float]:
        """
        Get all the metrics of the iteration with the minimum value of the given key.

        :param key: The key to get the minimum value of. If None, the first key is used.
        :type key: str

        :return: All the metrics of the iteration with the minimum value of the given key.
        :rtype: Dict[str, float]

        :raises ValueError: If the key is not in the container.
        """
        if key is None:
            key = list(self.keys())[0]
        if key in self:
            argmin = np.argmin(self[key])
            return {k: v[argmin] for k, v in self.items()}
        raise ValueError("key not in container")

    def max(self, key=None, default=-np.inf):
        """
        Get the maximum value of the given key.

        :param key: The key to get the maximum value of. If None, the first key is used.
        :type key: str
        :param default: The default value to return if the key is not in the container.
        :type default: float

        :return: The maximum value of the given key.
        :rtype: float
        """
        if key is None:
            key = list(self.keys())[0]
        if key in self and len(self[key]) > 0:
            return np.nanmax(self[key])
        return default

    def max_item(self, key=None):
        """
        Get all the metrics of the iteration with the maximum value of the given key.

        :param key: The key to get the maximum value of. If None, the first key is used.
        :type key: str

        :return: All the metrics of the iteration with the maximum value of the given key.
        :rtype: Dict[str, float]

        :raises ValueError: If the key is not in the container.
        """
        if key is None:
            key = list(self.keys())[0]
        if key in self:
            argmax = np.argmax(self[key])
            return {k: v[argmax] for k, v in self.items()}
        raise ValueError("key not in container")

    def get_item_at(self, idx: int = -1):
        """
        Get all the metrics of the iteration at the given index.

        :param idx: The index to get the metrics of.
        :type idx: int

        :return: All the metrics of the iteration at the given index.
        :rtype: Dict[str, float]

        :raises ValueError: If the index is out of bounds.
        """
        if idx >= len(self):
            raise ValueError("Index out of bounds")
        return {k: v[idx] for k, v in self.items()}

    def get(self, key, default=None):
        """
        Get the values of the given key.

        :param key: The key to get the values of.
        :type key: str
        :param default: The default value to return if the key is not in the container.
        :type default: Any

        :return: The values of the given key.
        :rtype: List[float]
        """
        return self._container.get(key, default)

    def create_plot(self, **kwargs) -> Tuple[plt.Figure, Dict[str, plt.Axes], Dict[str, plt.Line2D]]:
        """
        Create a plot of the metrics in the container.

        :param kwargs: Keyword arguments.

        :keyword Tuple[float, float] figsize: The size of the figure.
        :keyword int linewidth: The width of the lines.

        :return: The figure, axes and lines of the plot.
        :rtype: Tuple[plt.Figure, Dict[str, plt.Axes], Dict[str, plt.Line2D]]
        """
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
                    if k in keys_lower_to_given:
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
            save_path: Optional[str] = None,
            show: bool = False,
            **kwargs
    ) -> Tuple[plt.Figure, Dict[str, plt.Axes], Dict[str, plt.Line2D]]:
        """
        Plot the metrics in the container.

        :param save_path: The path to save the plot to. If None, the plot is not saved.
        :type save_path: Optional[str]

        :param show: Whether to show the plot.
        :type show: bool

        :param kwargs: Keyword arguments.

        :keyword Tuple[float, float] figsize: The size of the figure.
        :keyword int linewidth: The width of the lines.
        :keyword int dpi: The resolution of the figure.
        :keyword bool block: Whether to block execution until the plot is closed.
        :keyword bool close: Whether to close the plot at the end.

        :return: The figure, axes and lines of the plot.
        :rtype: Tuple[plt.Figure, Dict[str, plt.Axes], Dict[str, plt.Line2D]]
        """
        kwargs = self._set_default_plot_kwargs(kwargs)
        plt.close('all')
        fig, axes, lines = self.create_plot(**kwargs)
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
        """
        Update the plot of the metrics in the container.

        :param fig: The figure to update.
        :type fig: plt.Figure
        :param axes: The axes to update.
        :type axes: Dict[str, plt.Axes]
        :param lines: The lines to update.
        :type lines: Dict[str, plt.Line2D]
        :param kwargs: Keyword arguments.

        :return: The figure, axes and lines of the plot.
        :rtype: Tuple[plt.Figure, Dict[str, plt.Axes], Dict[str, plt.Line2D]]
        """
        kwargs = self._set_default_plot_kwargs(kwargs)
        for k in lines:
            lines[k].set_data(range(len(self[k])), self[k])
        for k in axes:
            axes[k].relim()
            axes[k].autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        return fig, axes, lines

    def on_iteration_end(self, trainer, **kwargs):
        """
        Insert the current metrics into the container.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        self.insert(
            trainer.current_training_state.iteration,
            {k: to_numpy(v) for k, v in trainer.current_training_state.itr_metrics.items()}
        )

    def extra_repr(self):
        return f", n={len(self)}, metrics={list(self.keys())}"







