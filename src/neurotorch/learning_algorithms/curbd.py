from collections import defaultdict
from typing import Optional, Sequence, Union, Dict, Callable, List, Tuple

import numpy as np
import torch

from .learning_algorithm import LearningAlgorithm
from ..learning_algorithms.tbptt import TBPTT
from ..transforms.base import ToDevice
from ..utils import compute_jacobian, list_insert_replace_at


class CURBD(TBPTT):
	r"""
	Apply the backpropagation through time algorithm to the given model.
	"""
	CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: str = "optimizer_state_dict"
	CHECKPOINT_P_STATE_DICT_KEY: str = "P"
	
	def __init__(
			self,
			*,
			params: Optional[Sequence[torch.nn.Parameter]] = None,
			layers: Optional[Union[Sequence[torch.nn.Module], torch.nn.Module]] = None,
			criterion: Optional[Union[Dict[str, Union[torch.nn.Module, Callable]], torch.nn.Module, Callable]] = None,
			backward_time_steps: Optional[int] = None,
			**kwargs
	):
		kwargs.setdefault("auto_backward_time_steps_ratio", 0)
		"""
		Constructor for WeakRLS class.

		:param params: The parameters to optimize. If None, the parameters of the model's trainer will be used.
		:type params: Optional[Sequence[torch.nn.Parameter]]
		:param criterion: The criterion to use. If not provided, torch.nn.MSELoss is used.
		:type criterion: Optional[Union[Dict[str, Union[torch.nn.Module, Callable]], torch.nn.Module, Callable]]
		:param kwargs: The keyword arguments to pass to the BaseCallback.

		:keyword bool save_state: Whether to save the state of the optimizer. Defaults to True.
		:keyword bool load_state: Whether to load the state of the optimizer. Defaults to True.
		"""
		kwargs.setdefault("save_state", True)
		kwargs.setdefault("load_state", True)
		super().__init__(
			params=params,
			layers=layers,
			criterion=criterion,
			backward_time_steps=backward_time_steps,
			optimizer=None,
			optim_time_steps=None,
			**kwargs
		)
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
		self.criterion = criterion
		
		# RLS attributes
		self.P = None
		self.P0 = kwargs.get("P0", 1.0)
		self._device = kwargs.get("device", None)
		self.to_cpu_transform = ToDevice(device=torch.device("cpu"))
		self.to_device_transform = None
		self.reduction = kwargs.get("reduction", "mean").lower()
		self._other_dims_as_batch = kwargs.get("other_dims_as_batch", False)
		self._is_recurrent = True
		
		self._asserts()
	
	def _asserts(self):
		assert self.reduction in ["mean", "sum", "none"], "reduction must be either 'mean', 'sum' or 'none'"
	
	def _initialize_P(self, m=None):
		self.P = [
			self.P0 * torch.eye(param.numel() if m is None else m, dtype=torch.float32, device=torch.device("cpu"))
			for param in self.params
		]
	
	def _maybe_update_time_steps(self):
		if self._auto_set_backward_time_steps:
			self.backward_time_steps = max(1, int(self._auto_backward_time_steps_ratio * self._data_n_time_steps))
	
	def _decorate_forward(self, forward, layer_name: str):
		def _forward(*args, **kwargs):
			out = forward(*args, **kwargs)
			t = kwargs.get("t", None)
			if t is None:
				return out
			out_tensor = self._get_out_tensor(out)
			list_insert_replace_at(self._layers_buffer[layer_name], t % self.backward_time_steps, out_tensor)
			if len(self._layers_buffer[layer_name]) == self.backward_time_steps and t != 0:
				self._backward_at_t(t, self.backward_time_steps, layer_name)
				out = self._detach_out(out)
			return out
		
		return _forward
	
	def _backward_at_t(self, t: int, backward_t: int, layer_name: str):
		y_batch = self._get_y_batch_slice_from_trainer((t + 1) - backward_t, t + 1, layer_name)
		pred_batch = self._get_pred_batch_from_buffer(layer_name)
		self._batch_step(pred_batch, y_batch)
		self._layers_buffer[layer_name].clear()
	
	def load_checkpoint_state(self, trainer, checkpoint: dict, **kwargs):
		if self.load_state:
			state = checkpoint.get(self.name, {})
			self.P = state.get(self.CHECKPOINT_P_STATE_DICT_KEY, None)
	
	def get_checkpoint_state(self, trainer, **kwargs) -> object:
		if self.save_state:
			return {
				self.CHECKPOINT_P_STATE_DICT_KEY: self.P,
			}
		return None
	
	def start(self, trainer, **kwargs):
		LearningAlgorithm.start(self, trainer, **kwargs)
		if not self.params and not self.layers:
			self.params = list(trainer.model.parameters())
		elif self.layers:
			self.params = [param for layer in self.layers for param in layer.parameters()]
			
		if len(self.params) > 1:
			raise NotImplementedError("CURBD does not support multiple parameters yet.")
		if self.params[0].ndim > 2:
			raise NotImplementedError("CURBD does not support parameters with more than 2 dimensions yet.")
		if self.params[0].shape[0] != self.params[0].shape[1]:
			raise NotImplementedError("CURBD does not support parameters that are not square yet.")
		
		self.optimizer = torch.optim.SGD(self.params, lr=1.0)
		
		if self.criterion is None and trainer.criterion is not None:
			self.criterion = trainer.criterion
		
		if self._device is None:
			self._device = trainer.model.device
		self.to_device_transform = ToDevice(device=self._device)
		self.output_layers: torch.nn.ModuleDict = trainer.model.output_layers
		self._initialize_original_forwards()
	
	def on_batch_begin(self, trainer, **kwargs):
		LearningAlgorithm.on_batch_begin(self, trainer, **kwargs)
		self.trainer = trainer
		if self._is_recurrent:
			self._data_n_time_steps = self._get_data_time_steps_from_y_batch(trainer.current_training_state.y_batch)
			self._maybe_update_time_steps()
			self.decorate_forwards()
	
	def on_batch_end(self, trainer, **kwargs):
		super().on_batch_end(trainer)
		self.undecorate_forwards()
		self._layers_buffer.clear()
		
	def zero_grad(self):
		# for param in self.params:
		# 	if param.grad is not None:
		# 		param.grad.detach_()
		# 		param.grad.zero_()
		if self.optimizer:
			self.optimizer.zero_grad()
	
	def _batch_step(self, pred_batch, y_batch):
		model_device = self.trainer.model.device
		assert isinstance(pred_batch, torch.Tensor), "pred_batch must be a torch.Tensor"
		assert isinstance(y_batch, torch.Tensor), "y_batch must be a torch.Tensor"
		assert pred_batch.shape[0] == y_batch.shape[0] == 1, \
			"pred_batch and y_batch must be of shape (1, ...). CURBD batch-wise not implemented yet."
		pred_batch_view, y_batch_view = pred_batch[:, -1].view(-1, 1), y_batch[:, -1].view(-1, 1)
		self.zero_grad()
		
		if self.P is None:
			self._initialize_P(m=y_batch_view.shape[0])
			if self.params[0].shape[-1] != self.P[0].shape[0]:
				raise NotImplementedError("CURBD does not support parameters that are not the same shape as the output yet.")
		self._try_put_on_device(self.trainer)
		
		error = self.to_device_transform(pred_batch_view - y_batch_view)
		K = [torch.matmul(self.P[i], pred_batch_view) for i in range(len(self.params))]  # (m, m) @ (m, B) -> (m, B)
		yPy = [torch.matmul(pred_batch_view.T, K[i]).item() for i in range(len(self.params))]  # (B, m) @ (m, B) -> (B, B)
		c = [1.0 / (1.0 + yPy[i]) for i in range(len(self.params))]  # (B, B)
		self.P = [self.P[i] - c[i] * torch.matmul(K[i], K[i].T) for i in range(len(self.params))]  # (m, m) - (B, B) * (m, B) @ (B, m) -> (m, m)?
		delta_w = [c[i] * torch.outer(error.flatten(), K[i].flatten()) for i in range(len(self.params))]    # (B, B) * (m * B) @ (m * B) -> (l, 1) ?
		# for i, param in enumerate(self.params):
		# 	param.data -= delta_w[i].to(param.device, non_blocking=True).view(param.data.shape).T
		for i, param in enumerate(self.params):
			param.grad = delta_w[i].to(param.device, non_blocking=True).view(param.data.shape).T.clone()
		self.optimizer.step()
		
		self._put_on_cpu()
		self.trainer.model.to(model_device, non_blocking=True)
	
	def _try_put_on_device(self, trainer):
		try:
			self.P = [self.to_device_transform(p) for p in self.P]
		except Exception as e:
			trainer.model = self.to_cpu_transform(trainer.model)
			self.P = [self.to_device_transform(p) for p in self.P]
	
	def _put_on_cpu(self):
		self.P = [self.to_cpu_transform(p) for p in self.P]
	
	def on_optimization_begin(self, trainer, **kwargs):
		y_batch = trainer.current_training_state.y_batch
		pred_batch = trainer.format_pred_batch(trainer.current_training_state.pred_batch, y_batch)
		
		if self._is_recurrent:
			for layer_name in self._layers_buffer:
				backward_t = len(self._layers_buffer[layer_name])
				if backward_t > 0:
					self._backward_at_t(self._data_n_time_steps - 1, backward_t, layer_name)
		else:
			self._batch_step(trainer, **kwargs)
		
		trainer.update_state_(batch_loss=self.apply_criterion(pred_batch, y_batch).detach_())
	
	def on_optimization_end(self, trainer, **kwargs):
		self.zero_grad()

	
	

