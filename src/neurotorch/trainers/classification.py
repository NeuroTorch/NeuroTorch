from typing import Callable, Dict, List, Optional, Union

import torch

from . import Trainer
from ..metrics import ClassificationMetrics


class ClassificationTrainer(Trainer):
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

	def apply_criterion_on_batch(
			self,
			x_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
			y_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
	) -> torch.Tensor:
		if self.model.training:
			pred, out, h_sates = self.model.get_prediction_log_proba(
				x_batch, re_outputs_trace=True, re_hidden_states=True
			)
		else:
			with torch.no_grad():
				pred, out, h_sates = self.model.get_prediction_log_proba(
					x_batch, re_outputs_trace=True, re_hidden_states=True
				)
		if isinstance(self.criterion, dict):
			if isinstance(y_batch, torch.Tensor):
				y_batch = {k: y_batch for k in self.criterion}
			if isinstance(pred, torch.Tensor):
				pred = {k: pred for k in self.criterion}
			assert isinstance(pred, dict) and isinstance(y_batch, dict) and isinstance(pred, dict), \
				"If criterion is a dict, pred, y_batch and pred must be a dict too."
			batch_loss = sum([
				self.criterion[k](pred[k], y_batch[k].long().to(self.device))
				for k in self.criterion
			])
		else:
			batch_loss = self.criterion(pred, y_batch.long().to(self.device))
		return batch_loss



