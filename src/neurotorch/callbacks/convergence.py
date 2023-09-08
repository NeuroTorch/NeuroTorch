import time
from typing import Optional

import numpy as np

from .base_callback import BaseCallback


class ConvergenceTimeGetter(BaseCallback):
    """
    Monitor the training process and return the time it took to pass the threshold.
    """
    def __init__(
            self,
            *,
            metric: str,
            threshold: float,
            minimize_metric: bool,
            **kwargs
    ):
        """
        Constructor for ConvergenceTimeGetter class.

        :param metric: Name of the metric to monitor.
        :type metric: str
        :param threshold: Threshold value for the metric.
        :type threshold: float
        :param minimize_metric: Whether to minimize or maximize the metric.
        :type minimize_metric: bool
        :param kwargs: The keyword arguments to pass to the BaseCallback.
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.metric = metric
        self.minimize_metric = minimize_metric
        self.threshold_met = False
        self.time_convergence = np.inf
        self.itr_convergence = np.inf
        self.training_time = np.inf
        self.start_time = None

    def load_checkpoint_state(self, trainer, checkpoint: dict, **kwargs):
        if self.save_state:
            state = checkpoint.get(self.name, {})
            if state.get("threshold") == self.threshold and state.get("metric") == self.metric:
                super().load_checkpoint_state(trainer, checkpoint)
                self.start_time = time.time()
                if np.isfinite(state.get("time_convergence")):
                    self.start_time -= state.get("training_time", 0)
        # TODO: change start time and add training time, etc.

    def start(self, trainer, **kwargs):
        super().start(trainer)
        self.start_time = time.time()

    def close(self, trainer, **kwargs):
        self.training_time = time.time() - self.start_time

    def on_iteration_end(self, trainer, **kwargs):
        if not self.threshold_met:
            if self.minimize_metric:
                self.threshold_met = trainer.current_training_state.itr_metrics[self.metric] < self.threshold
            else:
                self.threshold_met = trainer.current_training_state.itr_metrics[self.metric] > self.threshold
            if self.threshold_met:
                self.save_on(trainer, **kwargs)

    def save_on(self, trainer, **kwargs):
        self.time_convergence = time.time() - self.start_time
        self.itr_convergence = trainer.current_training_state.iteration
        for cm in trainer.checkpoint_managers:
            cm.save_on(trainer)

    def __repr__(self):
        repr_str = f"ConvergenceTimeGetter("
        repr_str += f"metric={self.metric}, "
        repr_str += f"threshold={self.threshold}, "
        repr_str += f"minimize_metric={self.minimize_metric})"
        repr_str += f"<time_convergence={self.time_convergence} [s], "
        repr_str += f"itr_convergence={self.itr_convergence}, "
        repr_str += f"training_time={self.training_time} [s]>"
        return repr_str



