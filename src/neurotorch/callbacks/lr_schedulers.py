from typing import List, Optional, Iterable, Union, Sequence

import numpy as np
import torch

from .base_callback import BaseCallback
from ..learning_algorithms.learning_algorithm import LearningAlgorithm


class LinearLRScheduler(BaseCallback):
    """
    This class is a callback that implements a linear learning rate decay. This is useful to decrease the learning rate
    over iterations. The learning rate is decreased linearly from lr_start to lr_end over n_steps iterations.

    :Attributes:
        - **lr_start** (float): The initial learning rate.
        - **lr_end** (float): The final learning rate.
        - **n_steps** (int): The number of steps over which the learning rate is decreased.
        - **lr** (float): The current learning rate.
        - **lr_decay** (float): The learning rate decay per step.
    """
    DEFAULT_LR_START = 1e-3
    DEFAULT_MIN_LR = 1e-12

    def __init__(
            self,
            lr_start: Union[float, Sequence[float]],
            lr_end: Union[float, Sequence[float]],
            n_steps: int,
            optimizer: Optional[torch.optim.Optimizer] = None,
            **kwargs
    ):
        """
        Construcor for the LinearLRScheduler class.

        :param lr_start: The initial learning rate. If a sequence is given, each entry of the sequence is used for each
            parameter group.
        :type lr_start: Union[float, Sequence[float]]
        :param lr_end: The final learning rate. If a sequence is given, each entry of the sequence is used for each
            parameter group.
        :type lr_end: Union[float, Sequence[float]]
        :param n_steps: The number of steps over which the learning rate is decreased.
        :type n_steps: int
        """
        super().__init__()
        self.lr_start = lr_start if isinstance(lr_start, Iterable) else [lr_start]
        self.lr_end = lr_end if isinstance(lr_end, Iterable) else [lr_end]
        self.n_steps = n_steps
        self.lr = self.lr_start
        self.lr_decay = [
            (_lr_start - _lr_end) / self.n_steps
            for _lr_start, _lr_end in zip(self.lr_start, self.lr_end)
        ]
        self.optimizer = optimizer
        self.log_lr_to_history = kwargs.get('log_lr_to_history', True)

    def _init_lr_decay(self):
        assert len(self.lr_start) == len(self.lr_end), \
            "The number of learning rates must match the number of minimum learning rates."
        for i, (_lr_start, _min_lr) in enumerate(zip(self.lr_start, self.lr_end)):
            _decay = (self.lr_start[i] - self.lr_end[i]) / self.n_steps
            if len(self.lr_decay) <= i:
                self.lr_decay.append(_decay)
            if self.lr_decay[i] is None:
                self.lr_decay[i] = _decay

    def start(self, trainer, **kwargs):
        """
        Initialize the learning rate of the optimizer and the :attr:`lr_start` attribute if necessary.

        :param trainer: The trainer object.
        :type trainer: Trainer

        :return: None
        """
        if self.optimizer is None:
            learning_algorithms = trainer.learning_algorithms
            for la in learning_algorithms:
                if isinstance(la, LearningAlgorithm) and hasattr(la, 'optimizer'):
                    self.optimizer = la.optimizer
                    break
            if self.optimizer is None:
                raise ValueError('No optimizer found in the callbacks of the trainer.')
        for i, g in enumerate(self.optimizer.param_groups):
            if len(self.lr_start) <= i:
                self.lr_start.append(g.get("lr", self.DEFAULT_LR_START))
            if self.lr_start[i] is None:
                self.lr_start[i] = g.get("lr", self.DEFAULT_LR_START)

        if len(self.lr_end) < len(self.lr_start):
            for i in range(len(self.lr_start)):
                if len(self.lr_end) <= i:
                    self.lr_end.append(self.DEFAULT_MIN_LR)
                if self.lr_end[i] is None:
                    self.lr_end[i] = self.DEFAULT_MIN_LR
        self._init_lr_decay()

    def on_iteration_end(self, trainer, **kwargs):
        """
        Decrease the learning rate linearly.

        :param trainer: The trainer object.
        :type trainer: Trainer

        :return: None
        """
        if self.log_lr_to_history:
            trainer.update_itr_metrics_state_(**{f'lr_{i}': _lr for i, _lr in enumerate(self.lr)})
        step = trainer.current_training_state.iteration
        self.lr = [
            max(_lr_start - step * self.lr_decay, _lr_end)
            for _lr_start, _lr_end in zip(self.lr_start, self.lr_end)
        ]
        assert len(self.lr) > 0, "No learning rate found."
        if len(self.lr) == 1:
            self.lr = [self.lr[0] for _ in self.optimizer.param_groups]
        assert len(self.lr) == len(self.optimizer.param_groups), \
            "The number of learning rates must match the number of parameter groups."
        for g, lr in zip(self.optimizer.param_groups, self.lr):
            g['lr'] = lr

    def extra_repr(self) -> str:
        return f"lr_start={self.lr_start}, lr_end={self.lr_end}, n_steps={self.n_steps}"


class LRSchedulerOnMetric(BaseCallback):
    """
    Class to schedule the learning rate of the optimizer based on the metric value.
    Each time the metric reach the next value of the schedule, the learning rate is multiplied by the given decay.
    The learning rate is also capped at the given minimum value.

    :Attributes:
        - **metric** (str): The metric to use to schedule the learning rate.
        - **metric_schedule** (Iterable[float]): The schedule of the metric.
        - **minimize_metric** (bool): Whether to minimize the metric or maximize it.
        - **lr_decay** (float): The decay factor to use when the metric reach the next value of the schedule.
        - **min_lr** (float): The minimum learning rate to use.
        - **lr_start** (float): The learning rate to use at the beginning of the training.
        - **lr** (float): The current learning rate.
        - **retain_progress** (bool): If True the current step of the scheduler will only increase when the metric reach
        the next value of the schedule. If False, the current step will increase or decrease depending on the metric.
        - **step** (int): The current step of the scheduler.
    """
    DEFAULT_LR_START = 1e-3
    DEFAULT_MIN_LR = 1e-12

    def __init__(
            self,
            metric: str,
            metric_schedule: Iterable[float],
            *,
            minimize_metric: Optional[bool] = None,
            lr_decay: Optional[Union[float, Sequence[float]]] = None,
            min_lr: Optional[Union[float, Sequence[float]]] = None,
            lr_start: Optional[Union[float, Sequence[float]]] = None,
            retain_progress: bool = True,
            optimizer: Optional[torch.optim.Optimizer] = None,
            **kwargs
    ):
        """
        Initialize the scheduler with the given metric and metric schedule.

        :param metric: The metric to use to schedule the learning rate.
        :type metric: str
        :param metric_schedule: The schedule of the metric.
        :type metric_schedule: Iterable[float]
        :param minimize_metric: Whether to minimize the metric or maximize it. If None, infer from the metric schedule.
        :type minimize_metric: Optional[bool]
        :param lr_decay: The decay factor to use when the metric reach the next value of the schedule. If None, the
        decay is computed automatically as (lr_start - min_lr) / len(metric_schedule).
        :type lr_decay: Optional[Union[float, Sequence[float]]]
        :param min_lr: The minimum learning rate to use.
        :type min_lr: Optional[Union[float, Sequence[float]]]
        :param lr_start: The learning rate to use at the beginning of the training. If None, the learning rate is
        get automatically as the learning rate of the first group of the optimizer.
        :type lr_start: Optional[Union[float, Sequence[float]]]
        :param retain_progress: If True the current step of the scheduler will only increase when the metric reach the
        next value of the schedule. If False, the current step will increase or decrease depending on the metric.
        :type retain_progress: bool
        :param optimizer: The optimizer whose learning rate will be scheduled. If None, the optimizer is get from the
            trainer. Note that in this case the first optimizer of the trainer's callbacks will be used.
        :type optimizer: Optional[torch.optim.Optimizer]

        :param kwargs: The keyword arguments to pass to the BaseCallback.

        :keyword bool log_lr_to_history: Whether to log the learning rate to the training history. Default: True.
        """
        super().__init__(**kwargs)
        self.metric = metric
        self.metric_schedule = np.asarray(metric_schedule)
        self._check_schedule_ascending_or_descending()
        self.minimize_metric = minimize_metric
        self._init_minimize_metric()
        self.lr_decay = lr_decay if isinstance(lr_decay, Sequence) else [lr_decay]
        self.min_lr = min_lr if isinstance(min_lr, Sequence) else [min_lr]
        self.lr_start = lr_start if isinstance(lr_start, Sequence) else [lr_start]
        self.lr = self.lr_start
        self.retain_progress = retain_progress
        self.step = 0
        self.optimizer = optimizer
        self.log_lr_to_history = kwargs.get('log_lr_to_history', True)

    def on_iteration_end(self, trainer, **kwargs):
        """
        Update the learning rate of the optimizer based on the metric value.

        :param trainer: The trainer object.
        :type trainer: Trainer

        :return: None
        """
        last_metric = trainer.training_history[self.metric][-1]
        if self.log_lr_to_history:
            trainer.update_itr_metrics_state_(**{f'lr_{i}': _lr for i, _lr in enumerate(self.lr)})
        self.step = self.update_step(last_metric)
        self.lr = [
            max(_lr_start - self.step * _lr_decay, _min_lr)
            for _lr_start, _lr_decay, _min_lr in zip(self.lr_start, self.lr_decay, self.min_lr)
        ]
        assert len(self.lr) > 0, "No learning rate found."
        if len(self.lr) == 1:
            self.lr = [self.lr[0] for _ in self.optimizer.param_groups]
        assert len(self.lr) == len(self.optimizer.param_groups), \
            "The number of learning rates must match the number of parameter groups."
        for g, lr in zip(self.optimizer.param_groups, self.lr):
            g['lr'] = lr

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
            self.minimize_metric = np.mean(np.diff(self.metric_schedule) <= 0) > 0.5

    def _init_lr_decay(self):
        assert len(self.lr_start) == len(self.min_lr), \
            "The number of learning rates must match the number of minimum learning rates."
        for i, (_lr_start, _min_lr) in enumerate(zip(self.lr_start, self.min_lr)):
            _decay = (self.lr_start[i] - self.min_lr[i]) / len(self.metric_schedule)
            if len(self.lr_decay) <= i:
                self.lr_decay.append(_decay)
            if self.lr_decay[i] is None:
                self.lr_decay[i] = _decay

    def update_step(self, last_metric: float) -> int:
        """
        Update the current step of the scheduler based on the metric value.

        :param last_metric: The last value of the metric.
        :type last_metric: float

        :return: The new step.
        :rtype: int
        """
        last_index = len(self.metric_schedule) - 1
        if self.minimize_metric:
            next_step = last_index - np.argmax((last_metric <= self.metric_schedule)[::-1])
        else:
            next_step = last_index - np.argmax((last_metric >= self.metric_schedule)[::-1])
        if self.retain_progress:
            next_step = max(self.step, next_step)
        self.step = next_step
        return self.step

    def start(self, trainer, **kwargs):
        """
        Initialize the learning rate of the optimizer and the :attr:`lr_start` attribute if necessary.

        :param trainer: The trainer object.
        :type trainer: Trainer

        :return: None
        """
        if self.optimizer is None:
            learning_algorithms = trainer.learning_algorithms
            for la in learning_algorithms:
                if isinstance(la, LearningAlgorithm) and hasattr(la, 'optimizer'):
                    self.optimizer = la.optimizer
                    break
            if self.optimizer is None:
                raise ValueError('No optimizer found in the callbacks of the trainer.')
        for i, g in enumerate(self.optimizer.param_groups):
            if len(self.lr_start) <= i:
                self.lr_start.append(g.get("lr", self.DEFAULT_LR_START))
            if self.lr_start[i] is None:
                self.lr_start[i] = g.get("lr", self.DEFAULT_LR_START)

        if len(self.min_lr) < len(self.lr_start):
            for i in range(len(self.lr_start)):
                if len(self.min_lr) <= i:
                    self.min_lr.append(self.DEFAULT_MIN_LR)
                if self.min_lr[i] is None:
                    self.min_lr[i] = self.DEFAULT_MIN_LR

        self.lr = self.lr_start
        self._init_lr_decay()

    def on_pbar_update(self, trainer, **kwargs) -> dict:
        """
        Return the learning rate to display in the progress bar.

        :param trainer: The trainer object.
        :type trainer: Trainer

        :return: The dictionary to update the progress bar.
        :rtype: dict
        """
        return {}

    def extra_repr(self) -> str:
        return f"metric={self.metric}, minimize={self.minimize_metric}"
