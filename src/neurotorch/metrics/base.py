from typing import Optional, Any, Dict, Union, List, Callable

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from neurotorch.modules import BaseModel


class BaseMetrics:
	METRICS_NAMES_SEP = ' '
	
	def __init__(
			self,
			model: BaseModel,
			metrics_names: Any = "all",
			device: Optional[torch.device] = None
	):
		"""

		:param model: the model to evaluate.
		:param metrics_names: the metrics to compute.
		:param device: The device in which the metrics will be computed.
		"""
		self.model = model
		self.metrics_names = self._format_metrics_names_(metrics_names)
		self._check_metrics_names_(self.metrics_names)
		self.device = device
	
	@property
	def metrics_functions(self) -> Dict[str, Callable]:
		all_metrics_names_to_func = BaseMetrics.get_unwrap_all_metrics_names_to_func()
		return {
			metric_name: all_metrics_names_to_func[metric_name]
			for metric_name in self.metrics_names
		}
	
	@staticmethod
	def _format_metrics_names_(metrics: Any) -> List[str]:
		if metrics == "all":
			return BaseMetrics.get_unique_metrics_names()
		if isinstance(metrics, str):
			metrics = metrics.split(BaseMetrics.METRICS_NAMES_SEP)
		elif isinstance(metrics, list):
			metrics = [metric.strip() for metric in metrics]
		else:
			raise ValueError("metrics must be a string or a list of strings")
		return metrics
	
	@staticmethod
	def _check_metrics_names_(metrics: List[str]) -> None:
		all_metrics_names = BaseMetrics.get_all_metrics_names()
		assert all([metric in all_metrics_names for metric in metrics]), \
			f"metrics must be in {all_metrics_names}"
	
	@staticmethod
	def get_all_metrics_names_to_func() -> Dict[str, Callable]:
		raise NotImplementedError()
	
	@staticmethod
	def get_unwrap_all_metrics_names_to_func() -> Dict[str, Callable]:
		return {
			metric_name: metric_func
			for metric_names, metric_func in BaseMetrics.get_all_metrics_names_to_func().items()
			for metric_name in metric_names.split(BaseMetrics.METRICS_NAMES_SEP)
		}
	
	@staticmethod
	def get_all_metrics_names() -> List[str]:
		all_metrics_names = []
		for metric_names, _ in BaseMetrics.get_all_metrics_names_to_func().items():
			all_metrics_names.extend(metric_names.split(BaseMetrics.METRICS_NAMES_SEP))
		return all_metrics_names
	
	@staticmethod
	def get_unique_metrics_names() -> List[str]:
		all_metrics_names = []
		for metric_names, _ in BaseMetrics.get_all_metrics_names_to_func().items():
			all_metrics_names.extend(metric_names.split(BaseMetrics.METRICS_NAMES_SEP)[0])
		return all_metrics_names
	
	def __call__(
			self,
			data_loader: DataLoader,
			verbose: Union[bool, int] = False
	) -> Dict[str, Any]:
		"""
		Compute the metrics for the given data_loader.
		:param data_loader: The data loader to use to compute the metrics.
		:param verbose: 0: no progress bar, 1: single progress bar, 2: progress bar for each metrics
						True: verbose is set to 1, False: verbose is set to 0.
		:return: The metrics computed.
		"""
		if isinstance(verbose, bool):
			verbose = 1 if verbose else 0
		assert verbose in [0, 1, 2], "verbose must be 0, 1 or 2"
		output = {}
		p_bar = tqdm(
			enumerate(self.metrics_functions.items()),
			total=len(self.metrics_functions),
			disable=verbose == 0,
			unit="metric",
			position=0,
		)
		for i, (metric_name, metric_func) in p_bar:
			output[metric_name] = metric_func(
				model=self.model,
				dataloader=data_loader,
				device=self.device,
				verbose=verbose == 2,
				desc=metric_name,
				p_bar_position=i + 1
			)
		return output














