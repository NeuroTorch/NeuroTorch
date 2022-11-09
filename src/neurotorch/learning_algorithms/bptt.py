from typing import Optional, Sequence, Union, Dict, Callable, List

import torch

from .learning_algorithm import LearningAlgorithm


class BPTT(LearningAlgorithm):
	r"""
	Apply the backpropagation through time algorithm to the given model.
	"""
	CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: str = "optimizer_state_dict"
	
	def __init__(
			self,
			*,
			params: Optional[Sequence[torch.nn.Parameter]] = None,
			layers: Optional[Union[Sequence[torch.nn.Module], torch.nn.Module]] = None,
			optimizer: Optional[torch.optim.Optimizer] = None,
			criterion: Optional[Union[Dict[str, Union[torch.nn.Module, Callable]], torch.nn.Module, Callable]] = None,
			**kwargs
	):
		"""
		Constructor for BPTT class.
		
		:param params: The parameters to optimize. If None, the parameters of the model's trainer will be used.
		:type params: Optional[Sequence[torch.nn.Parameter]]
		:param optimizer: The optimizer to use. If not provided, torch.optim.Adam is used.
		:type optimizer: Optional[torch.optim.Optimizer]
		:param criterion: The criterion to use. If not provided, torch.nn.MSELoss is used.
		:type criterion: Optional[Union[Dict[str, Union[torch.nn.Module, Callable]], torch.nn.Module, Callable]]
		:param kwargs: The keyword arguments to pass to the BaseCallback.
		
		:keyword bool save_state: Whether to save the state of the optimizer. Defaults to True.
		:keyword bool load_state: Whether to load the state of the optimizer. Defaults to True.
		"""
		kwargs.setdefault("save_state", True)
		kwargs.setdefault("load_state", True)
		super().__init__(params=params, **kwargs)
		if params is None:
			params = []
		else:
			params = list(params)
		if layers is not None:
			if isinstance(layers, torch.nn.Module):
				layers = [layers]
			params.extend([param for layer in layers for param in layer.parameters() if param not in params])
		self.params: List[torch.nn.Parameter] = params
		self.layers = layers
		self.optimizer = optimizer
		self.criterion = criterion
		
	def load_checkpoint_state(self, trainer, checkpoint: dict, **kwargs):
		if self.save_state:
			state = checkpoint.get(self.name, {})
			opt_state_dict = state.get(self.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY, None)
			if opt_state_dict is not None:
				self.optimizer.load_state_dict(opt_state_dict)
			
	def get_checkpoint_state(self, trainer, **kwargs) -> object:
		if self.save_state:
			if self.optimizer is not None:
				return {
					self.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict()
				}
		return None
	
	def start(self, trainer, **kwargs):
		super().start(trainer)
		if self.params and self.optimizer is None:
			self.optimizer = torch.optim.Adam(self.params)
		elif not self.params and self.optimizer is not None:
			self.params.extend([
				param
				for i in range(len(self.optimizer.param_groups))
				for param in self.optimizer.param_groups[i]["params"]
			])
		else:
			self.params = trainer.model.parameters()
			self.optimizer = torch.optim.Adam(self.params)
		
		if self.criterion is None and trainer.criterion is not None:
			self.criterion = trainer.criterion
	
	def apply_criterion(self, pred_batch, y_batch):
		if self.criterion is None:
			if isinstance(y_batch, dict):
				self.criterion = {key: torch.nn.MSELoss() for key in y_batch}
			else:
				self.criterion = torch.nn.MSELoss()
		
		if isinstance(self.criterion, dict):
			if isinstance(y_batch, torch.Tensor):
				y_batch = {k: y_batch for k in self.criterion}
			if isinstance(pred_batch, torch.Tensor):
				pred_batch = {k: pred_batch for k in self.criterion}
			assert isinstance(pred_batch, dict) and isinstance(y_batch, dict), \
				"If criterion is a dict, pred, y_batch and pred must be a dict too."
			batch_loss = sum([
				self.criterion[k](pred_batch[k], y_batch[k].to(pred_batch[k].device))
				for k in self.criterion
			])
		else:
			if isinstance(pred_batch, dict) and len(pred_batch) == 1:
				pred_batch = pred_batch[list(pred_batch.keys())[0]]
			batch_loss = self.criterion(pred_batch, y_batch.to(pred_batch.device))
		return batch_loss
	
	def _make_optim_step(self, pred_batch, y_batch, retain_graph=False):
		self.optimizer.zero_grad()
		batch_loss = self.apply_criterion(pred_batch, y_batch)
		batch_loss.backward(retain_graph=retain_graph)
		self.optimizer.step()
		return batch_loss.detach_()
	
	def on_optimization_begin(self, trainer, **kwargs):
		y_batch = trainer.current_training_state.y_batch
		pred_batch = trainer.format_pred_batch(trainer.current_training_state.pred_batch, y_batch)
		batch_loss = self._make_optim_step(pred_batch, y_batch)
		trainer.update_state_(batch_loss=batch_loss)
	
	def on_optimization_end(self, trainer, **kwargs):
		self.optimizer.zero_grad()
	
	def on_validation_batch_begin(self, trainer, **kwargs):
		y_batch = trainer.current_training_state.y_batch
		pred_batch = trainer.format_pred_batch(trainer.current_training_state.pred_batch, y_batch)
		batch_loss = self.apply_criterion(pred_batch, y_batch)
		trainer.update_state_(batch_loss=batch_loss)
		
	def extra_repr(self) -> str:
		return f"optimizer={self.optimizer}, criterion={self.criterion}"

