from typing import Callable, Optional

from .base_callback import BaseCallback


class EventOnMetricThreshold(BaseCallback):
    r"""
    This class is a callback that call an event if the metric value reach a given threshold at the end of each
    iteration. Note that the trainer will be passed as the first argument to the event. In addition, if the
    given metric is not present in the training state, the event will not be called. Finally, the output of the event
    will be pass to the :meth:`on_pbar_update` method.

    :Attributes:
        - **metric_name** (str): The name of the metric to monitor.
        - **threshold** (float): The threshold value.
        - **event** (Callable): The event to call.
        - **event_args** (tuple): The arguments to pass to the event.
        - **event_kwargs** (dict): The keyword arguments to pass to the event.
    """
    def __init__(
            self,
            metric_name: str,
            threshold: float,
            *,
            event: Callable,
            event_args: Optional[tuple] = None,
            event_kwargs: Optional[dict] = None,
            minimize_metric: bool = True,
            do_once: bool = False,
            **kwargs
    ):
        """
        Constructor for the EventOnMetricThreshold class.

        :param metric_name: The name of the metric to monitor.
        :type metric_name: str
        :param threshold: The threshold value.
        :type threshold: float
        :param event: The event to call.
        :type event: Callable
        :param event_args: The arguments to pass to the event.
        :type event_args: tuple
        :param event_kwargs: The keyword arguments to pass to the event.
        :type event_kwargs: dict
        :param kwargs: The keyword arguments to pass to the base class.
        """
        super().__init__(**kwargs)
        self.metric_name = metric_name
        self.threshold = threshold
        self.event = event
        self.event_args = event_args if event_args is not None else tuple()
        self.event_kwargs = event_kwargs if event_kwargs is not None else dict()
        self._current_event_output = None
        self.minimize_metric = minimize_metric
        self.do_once = do_once
        self._has_triggered = False

    def on_iteration_end(self, trainer, **kwargs):
        """
        Check if the metric value reach the threshold.

        :param trainer: The trainer object.
        :type trainer: Trainer

        :return: None
        """
        if self.do_once and self._has_triggered:
            return
        if self.metric_name in trainer.current_training_state.itr_metrics:
            metric_value = trainer.current_training_state.itr_metrics[self.metric_name]
            if self.minimize_metric:
                threshold_reached = metric_value <= self.threshold
            else:
                threshold_reached = metric_value >= self.threshold
            if threshold_reached:
                self._current_event_output = self.event(trainer, *self.event_args, **self.event_kwargs)
                self._has_triggered = True

    def on_pbar_update(self, trainer, **kwargs) -> dict:
        """
        Update the progress bar with the current learning algorithm.

        :param trainer: The trainer object.
        :type trainer: Trainer

        :return: The progress bar update.
        :rtype: dict
        """
        if self._current_event_output is not None:
            return {self.name: self._current_event_output}
        return dict()




