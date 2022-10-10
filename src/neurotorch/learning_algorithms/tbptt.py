from collections import defaultdict
from typing import Optional, Sequence, Union, Dict, Callable, List, Tuple

import torch
from .bptt import BPTT


class TBPTT(BPTT):
	def __init__(
			self,
			*,
			params: Optional[Sequence[torch.nn.Parameter]] = None,
			layers: Optional[Union[Sequence[torch.nn.Module], torch.nn.Module]] = None,
			optimizer: Optional[torch.optim.Optimizer] = None,
			criterion: Optional[Union[Dict[str, Union[torch.nn.Module, Callable]], torch.nn.Module, Callable]] = None,
			backward_time_steps: Optional[int] = None,
			optim_time_steps: Optional[int] = None,
			**kwargs
	):
		super(TBPTT, self).__init__(params=params, layers=layers, optimizer=optimizer, criterion=criterion, **kwargs)
		self.output_layers = None
		self.backward_time_steps = backward_time_steps
		self._auto_backward_time_steps_ratio = kwargs.get("auto_backward_time_steps_ratio", 0.1)
		self.optim_time_steps = optim_time_steps
		self._auto_optim_time_steps_ratio = kwargs.get("auto_optim_time_steps_ratio", 0.1)
		self._max_t = 0
		self._layers_buffer = defaultdict(list)
	
	def load_checkpoint_state(self, trainer, checkpoint: dict):
		if self.save_state:
			state = checkpoint.get(self.name, {})
			opt_state_dict = state.get(self.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY, None)
			if opt_state_dict is not None:
				self.optimizer.load_state_dict(opt_state_dict)
	
	def get_checkpoint_state(self, trainer) -> object:
		if self.save_state:
			if self.optimizer is not None:
				return {
					self.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict()
				}
		return None
	
	def start(self, trainer):
		super().start(trainer)
		self.output_layers: torch.nn.ModuleDict = trainer.model.output_layers
		self.decorate_forwards()
	
	def decorate_forwards(self):
		for layer in self.output_layers.values():
			layer.forward = self._decorate_forward(layer.forward, layer.name)
	
	def _maybe_update_time_steps(self, t: int) -> bool:
		time_steps_are_valid = True
		if self.backward_time_steps is None:
			if t < self._max_t:
				self.backward_time_steps = int(max(self._auto_backward_time_steps_ratio * self._max_t, 1))
				self.optim_time_steps = self.backward_time_steps
			else:
				time_steps_are_valid = False
		if self.optim_time_steps is None:
			if t < self._max_t:
				self.optim_time_steps = int(max(self._auto_optim_time_steps_ratio * self._max_t, 1))
			else:
				time_steps_are_valid = False
		return time_steps_are_valid
	
	def _decorate_forward(self, forward, layer_name: str):
		def _forward(*args, **kwargs):
			t = kwargs.get("t", None)
			self._max_t = max(self._max_t, t)
			out = forward(*args, **kwargs)
			if t is None or not self.trainer.model.training:
				return out
			if not self._maybe_update_time_steps(t):
				return out
			if (t+1) % self.backward_time_steps == 0:
				y_batch = self._get_y_batch_slice_from_trainer((t+1) - self.backward_time_steps, t + 1, layer_name)
				pred_batch = self._get_pred_batch_from_buffer(layer_name)
				self._make_optim_step(pred_batch, y_batch, retain_graph=True)
				# batch_loss = self.apply_criterion(pred_batch, y_batch)
				# batch_loss.backward(retain_graph=True)
				# batch_loss.detach_()
				self._layers_buffer[layer_name].clear()
			out_tensor = self._get_out_tensor(out)
			self._layers_buffer[layer_name].append(out_tensor)
			return out
		return _forward
	
	def _get_y_batch_slice_from_trainer(self, t_first: int, t_last: int, layer_name: str = None):
		y_batch = self.trainer.current_training_state.y_batch
		if isinstance(y_batch, dict):
			if layer_name is None:
				y_batch = {
					key: val[:, t_first:t_last]
					for key, val in y_batch.items()
				}
			else:
				y_batch = y_batch[layer_name][:, t_first:t_last]
		else:
			y_batch = y_batch[:, t_first:t_last]
		return y_batch.clone()
	
	def _get_out_tensor(self, out: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]]):
		if isinstance(out, (tuple, list)):
			out = out[0]
		return out
	
	def _get_pred_batch_from_buffer(self, layer_name: str):
		pred_batch = torch.stack(self._layers_buffer[layer_name], dim=1)
		return pred_batch


