from typing import Optional, Dict, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from neurotorch.metrics.base import BaseMetrics
from neurotorch.modules import BaseModel


class ClassificationMetrics(BaseMetrics):
	@staticmethod
	def get_all_metrics_names_to_func() -> Dict[str, Callable]:
		sep = ClassificationMetrics.METRICS_NAMES_SEP
		return {
			f"accuracy{sep}acc": ClassificationMetrics.accuracy,
		}
	
	@staticmethod
	def accuracy(
			model: BaseModel,
			dataloader: DataLoader,
			device: Optional[torch.device] = None,
			verbose: bool = False,
			desc: Optional[str] = None,
			p_bar_position: int = 0,
	) -> float:
		""" Computes classification accuracy on supplied data in batches. """
		if device is not None:
			model.to(device)
		model.eval()
		accs = []
		with torch.no_grad():
			for i, (inputs, classes) in tqdm(
					enumerate(dataloader), total=len(dataloader),
					desc=desc, disable=not verbose, position=p_bar_position,
			):
				inputs = inputs.to(model.device)
				classes = classes.to(model.device)
				outputs = model.get_prediction_logits(inputs, re_outputs_trace=False, re_hidden_states=False)
				_, preds = torch.max(outputs, -1)
				accs.extend(torch.eq(preds, classes).float().cpu().numpy())
		return np.mean(np.asarray(accs)).item()
