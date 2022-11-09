from typing import Callable, Dict, List, Optional, Union

import torch

from . import Trainer
from ..transforms.base import ToTensor
from ..metrics import ClassificationMetrics


class ClassificationTrainer(Trainer):
	def __init__(self, *args, **kwargs):
		kwargs.setdefault("y_transform", ToTensor(dtype=torch.long))
		kwargs.setdefault("predict_method", "get_prediction_log_proba")
		super().__init__(*args, **kwargs)
	
	def _set_default_criterion(self, criterion: Optional[torch.nn.Module]) -> torch.nn.Module:
		if criterion is None:
			if isinstance(self.model.output_sizes, dict):
				criterion = {
					k: torch.nn.NLLLoss() for k in self.model.output_sizes
				}
			elif isinstance(self.model.output_sizes, int):
				criterion = torch.nn.NLLLoss()
			else:
				raise ValueError("Unknown criterion type")
		return criterion

	def _set_default_metrics(self, metrics: Optional[List[Callable]]):
		if metrics is None:
			metrics = [ClassificationMetrics(self.model)]
		return metrics



