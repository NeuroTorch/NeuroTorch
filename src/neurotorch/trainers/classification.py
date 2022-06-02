from typing import Optional

import torch

from . import Trainer


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
		
	def apply_criterion_on_batch(self, x_batch, y_batch):
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
			if len(self.criterion) == 1 and isinstance(pred, torch.Tensor) and isinstance(y_batch, torch.Tensor):
				return list(self.criterion.values())[0](pred, y_batch.long().to(self.device))
			assert isinstance(x_batch, dict) and isinstance(y_batch, dict) and isinstance(pred, dict), \
				"If criterion is a dict, x_batch, y_batch and pred must be a dict too."
			batch_loss = sum([
				self.criterion[k](pred[k], y_batch[k].long().to(self.device))
				for k in self.criterion
			])
		else:
			batch_loss = self.criterion(pred, y_batch.long().to(self.device))
		return batch_loss



