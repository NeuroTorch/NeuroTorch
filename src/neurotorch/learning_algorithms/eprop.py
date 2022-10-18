import warnings
from typing import Optional, Sequence, Union, Dict, Callable

import torch

from .learning_algorithm import LearningAlgorithm
from ..utils import batchwise_temporal_filter


class Eprop(LearningAlgorithm):
	r"""
	Apply the eligibility trace forward propagation (e-prop) :cite:t:`bellec_solution_2020`
	algorithm to the given model.
	"""
	CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: str = "optimizer_state_dict"
	CHECKPOINT_RN_FEEDBACK_WEIGHTS_KEY: str = "rn_feedback_weights"
	
	def __init__(
			self,
			*,
			params: Optional[Sequence[torch.nn.Parameter]] = None,
			optimizer: Optional[torch.optim.Optimizer] = None,
			criterion: Optional[Union[Dict[str, Union[torch.nn.Module, Callable]], torch.nn.Module, Callable]] = None,
			**kwargs
	):
		"""
		Constructor for Eprop class.

		:param params: The parameters to optimize. If None, the parameters of the model's trainer will be used.
		:type params: Optional[Sequence[torch.nn.Parameter]]
		:param optimizer: The optimizer to use. If not provided, torch.optim.SGD is used.
		:type optimizer: Optional[torch.optim.Optimizer]
		:param criterion: The criterion to use. If not provided, torch.nn.MSELoss is used.
		:type criterion: Optional[Union[Dict[str, Union[torch.nn.Module, Callable]], torch.nn.Module, Callable]]
		:param kwargs: The keyword arguments to pass to the BaseCallback.

		:keyword bool save_state: Whether to save the state of the optimizer. Defaults to True.
		:keyword bool load_state: Whether to load the state of the optimizer. Defaults to True.
		"""
		warnings.warn("Eprop is still in beta and may not work as expected or act exactly as BPTT.")
		kwargs.setdefault("save_state", True)
		kwargs.setdefault("load_state", True)
		super().__init__(**kwargs)
		self.params = params
		self.optimizer = optimizer
		self.criterion = criterion
		self.random_feedbacks = kwargs.get("random_feedbacks", True)
		if not self.random_feedbacks:
			raise NotImplementedError("Non-random feedbacks are not implemented yet.")
		self.rn_feedback_weights = None
		self.rn_gen = torch.Generator()
		self.rn_gen.manual_seed(kwargs.get("seed", 0))
	
	def load_checkpoint_state(self, trainer, checkpoint: dict, **kwargs):
		if self.save_state:
			state = checkpoint.get(self.name, {})
			opt_state_dict = state.get(self.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY, None)
			if opt_state_dict is not None:
				self.optimizer.load_state_dict(opt_state_dict)
			if self.random_feedbacks:
				self.rn_feedback_weights = state.get(self.CHECKPOINT_RN_FEEDBACK_WEIGHTS_KEY, None)
	
	def get_checkpoint_state(self, trainer, **kwargs) -> object:
		if self.save_state:
			state = {}
			if self.optimizer is not None:
				state[self.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY] = self.optimizer.state_dict()
			if self.random_feedbacks:
				state[self.CHECKPOINT_RN_FEEDBACK_WEIGHTS_KEY] = self.rn_feedback_weights
			return state
		return None
	
	def start(self, trainer, **kwargs):
		super().start(trainer)
		if self.params is not None and self.optimizer is None:
			self.optimizer = torch.optim.SGD(self.params, lr=1e-3)
		elif self.params is None and self.optimizer is not None:
			self.params = [
				param
				for i in range(len(self.optimizer.param_groups))
				for param in self.optimizer.param_groups[i]["params"]
			]
		else:
			self.params = trainer.model.parameters()
			self.optimizer = torch.optim.SGD(self.params, lr=1e-3)
		
		if self.criterion is None and trainer.criterion is not None:
			self.criterion = trainer.criterion
	
	def get_learning_signals(self, trainer, **kwargs):
		y_batch = trainer.current_training_state.y_batch
		pred_batch = trainer.format_pred_batch(trainer.current_training_state.pred_batch, y_batch)
		
		if self.rn_feedback_weights is None:
			self.rn_feedback_weights = [
				torch.randn((pred_batch.shape[-1], w.shape[-1]), generator=self.rn_gen)
				for w in self.params
			]
		
		if isinstance(y_batch, dict):
			if isinstance(pred_batch, torch.Tensor):
				pred_batch = {k: pred_batch for k in y_batch}
			assert isinstance(pred_batch, dict) and isinstance(y_batch, dict), \
				"If y_batch is a dict, pred must be a dict too."
			batch_err = sum([
					(pred_batch[k] - y_batch[k].to(pred_batch[k].device))
					for k in y_batch
			])
		else:
			if isinstance(pred_batch, dict) and len(pred_batch) == 1:
				pred_batch = pred_batch[list(pred_batch.keys())[0]]
			batch_err = (pred_batch - y_batch.to(pred_batch.device))
		
		batch_learning_signals = [torch.matmul(batch_err, B) for B in self.rn_feedback_weights]
		return batch_learning_signals
	
	def get_eligibility_trace(self, trainer, **kwargs):
		y_batch = trainer.current_training_state.y_batch
		pred_batch = trainer.format_pred_batch(trainer.current_training_state.pred_batch, y_batch)
		eligibility_vectors = batchwise_temporal_filter(pred_batch)
		return eligibility_vectors
	
	def on_optimization_begin(self, trainer, **kwargs):
		self.optimizer.zero_grad()
		learning_signals = self.get_learning_signals(trainer, **kwargs)
		eligibility_traces = self.get_eligibility_trace(trainer, **kwargs)
		self.optimizer.step()
	
	def on_optimization_end(self, trainer, **kwargs):
		self.optimizer.zero_grad()
	
