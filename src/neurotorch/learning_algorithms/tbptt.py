import warnings
from collections import defaultdict
from typing import Optional, Sequence, Union, Dict, Callable, List, Tuple

import torch
from .bptt import BPTT
from ..utils import list_insert_replace_at, zero_grad_params, recursive_detach, unpack_out_hh


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
		self._original_forwards = {}  # {layer_name: (layer, layer.forward)}
		self._auto_set_backward_time_steps = backward_time_steps is None
		self.backward_time_steps = backward_time_steps
		self._auto_backward_time_steps_ratio = kwargs.get("auto_backward_time_steps_ratio", 0.1)
		assert 0 <= self._auto_backward_time_steps_ratio <= 1, "auto_backward_time_steps_ratio must be between 0 and 1"
		self._auto_set_optim_time_steps = optim_time_steps is None
		self.optim_time_steps = optim_time_steps
		self._auto_optim_time_steps_ratio = kwargs.get("auto_optim_time_steps_ratio", self._auto_backward_time_steps_ratio)
		assert 0 <= self._auto_optim_time_steps_ratio <= 1, "auto_optim_time_steps_ratio must be between 0 and 1"
		self._data_n_time_steps = 0
		self._layers_buffer = defaultdict(list)
		self._forwards_decorated = False
		self._optim_counter = 0
	
	def start(self, trainer, **kwargs):
		super().start(trainer)
		self.output_layers: torch.nn.ModuleDict = trainer.model.output_layers
		self._initialize_original_forwards()
		
	def on_batch_begin(self, trainer, **kwargs):
		super().on_batch_begin(trainer)
		self.trainer = trainer
		if trainer.model.training:
			self._data_n_time_steps = self._get_data_time_steps_from_y_batch(trainer.current_training_state.y_batch)
			self._maybe_update_time_steps()
			self.optimizer.zero_grad()
			self.decorate_forwards()
	
	def on_batch_end(self, trainer, **kwargs):
		super().on_batch_end(trainer)
		if trainer.model.training:
			for layer_name in self._layers_buffer:
				backward_t = len(self._layers_buffer[layer_name])
				if backward_t > 0:
					self._backward_at_t(self._data_n_time_steps - 1, backward_t, layer_name)
					self.optimizer.step()
		self.undecorate_forwards()
		self._layers_buffer.clear()
		self.optimizer.zero_grad()
	
	def _get_data_time_steps_from_y_batch(self, y_batch: Union[torch.Tensor, Dict[str, torch.Tensor]]):
		if isinstance(y_batch, torch.Tensor):
			return y_batch.shape[-2]
		elif isinstance(y_batch, dict):
			return max([y.shape[-2] for y in y_batch.values()])
		else:
			raise ValueError(f"y_batch must be either a torch.Tensor or a dict, but got {type(y_batch)}")
	
	def _initialize_original_forwards(self):
		for layer in self.output_layers.values():
			self._original_forwards[layer.name] = (layer, layer.forward)
	
	def decorate_forwards(self):
		if self.trainer.model.training:
			if not self._forwards_decorated:
				self._initialize_original_forwards()
			for layer in self.output_layers.values():
				layer.forward = self._decorate_forward(layer.forward, layer.name)
			self._forwards_decorated = True
	
	def undecorate_forwards(self):
		for name, (layer, original_forward) in self._original_forwards.items():
			layer.forward = original_forward
		self._forwards_decorated = False
	
	def _maybe_update_time_steps(self):
		if self._auto_set_backward_time_steps:
			self.backward_time_steps = max(1, int(self._auto_backward_time_steps_ratio * self._data_n_time_steps))
		if self._auto_set_optim_time_steps:
			self.optim_time_steps = max(1, int(self._auto_optim_time_steps_ratio * self._data_n_time_steps))
		# if self.backward_time_steps != self.optim_time_steps:
		# 	raise NotImplementedError("backward_time_steps != optim_time_steps is not implemented yet")
		if self.backward_time_steps > self.optim_time_steps:
			raise NotImplementedError("backward_time_steps must be lower or equal to optim_time_steps.")
	
	def _decorate_forward(self, forward, layer_name: str):
		def _forward(*args, **kwargs):
			out = forward(*args, **kwargs)
			t, forecasting = kwargs.get("t", None), kwargs.get("forecasting", False)
			if t is None:
				return out
			out_tensor, hh = unpack_out_hh(out)
			list_insert_replace_at(self._layers_buffer[layer_name], t % self.backward_time_steps, out_tensor)
			self._optim_counter += 1
			if len(self._layers_buffer[layer_name]) == self.backward_time_steps:
				self._backward_at_t(t, self.backward_time_steps, layer_name)
				out = recursive_detach(out)
			if self._optim_counter >= self.optim_time_steps:
				self._make_optim_step()
			return out
		return _forward
	
	def _backward_at_t(self, t: int, backward_t: int, layer_name: str):
		y_batch = self._get_y_batch_slice_from_trainer((t + 1) - backward_t, t + 1, layer_name)
		pred_batch = self._get_pred_batch_from_buffer(layer_name)
		batch_loss = self.apply_criterion(pred_batch, y_batch)
		if batch_loss.grad_fn is None:
			raise ValueError(
				f"batch_loss.grad_fn is None. This is probably an internal error. Please report this issue on GitHub."
			)
		batch_loss.backward()
		self._layers_buffer[layer_name].clear()
		
	def _make_optim_step(self, **kwargs):
		self.optimizer.step()
		self.optimizer.zero_grad()
		zero_grad_params(self.params)
		self._optim_counter = 0
	
	def _get_y_batch_slice_from_trainer(self, t_first: int, t_last: int, layer_name: str = None):
		y_batch = self.trainer.current_training_state.y_batch.clone()
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
		return y_batch

	def _get_pred_batch_from_buffer(self, layer_name: str):
		pred_batch = torch.stack(self._layers_buffer[layer_name], dim=1)
		return pred_batch
	
	def on_optimization_begin(self, trainer, **kwargs):
		y_batch = trainer.current_training_state.y_batch
		pred_batch = trainer.format_pred_batch(trainer.current_training_state.pred_batch, y_batch)
		batch_loss = self.apply_criterion(pred_batch, y_batch)
		trainer.update_state_(batch_loss=batch_loss)
	
	def on_optimization_end(self, trainer, **kwargs):
		super(TBPTT, self).on_optimization_end(trainer)
		self._layers_buffer.clear()
	
	def close(self, trainer, **kwargs):
		self.undecorate_forwards()
		super(TBPTT, self).close(trainer)

