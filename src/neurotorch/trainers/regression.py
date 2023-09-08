from typing import Callable, Dict, List, Optional, Union, Any

import torch

from . import Trainer
from ..metrics import RegressionMetrics


class RegressionTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("predict_method", "get_prediction_trace")
        super().__init__(*args, **kwargs)

    @staticmethod
    def _set_default_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = Trainer._set_default_kwargs(kwargs)
        kwargs.setdefault("foresight_time_steps", None)
        return kwargs

    def _set_default_criterion(self, criterion: Optional[torch.nn.Module]) -> torch.nn.Module:
        if criterion is None:
            if isinstance(self.model.output_sizes, dict):
                criterion = {
                    k: torch.nn.MSELoss() for k in self.model.output_sizes
                }
            elif isinstance(self.model.output_sizes, int):
                criterion = torch.nn.MSELoss()
            else:
                raise ValueError("Unknown criterion type")
        return criterion

    def _set_default_metrics(self, metrics: Optional[List[Callable]]):
        if metrics is None:
            metrics = [RegressionMetrics(self.model)]
        return metrics


