import warnings
from collections import defaultdict
from typing import Optional, Sequence, Union, Dict, Callable, Tuple, List

import torch
from unstable import unstable

from .learning_algorithm import LearningAlgorithm
from ..transforms.base import to_numpy
from ..learning_algorithms.tbptt import TBPTT
from ..utils import batchwise_temporal_filter, list_insert_replace_at, zero_grad_params


@unstable
class Eprop(TBPTT):
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
			layers: Optional[Union[Sequence[torch.nn.Module], torch.nn.Module]] = None,
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
		warnings.warn("Eprop is still in beta and may not work as expected or act exactly as BPTT.", DeprecationWarning)
		kwargs.setdefault("save_state", True)
		kwargs.setdefault("load_state", True)
		# TODO: implement a optim step at each `optim_time_steps` steps.
		assert "backward_time_steps" not in kwargs, f"{self.__class__} does not support backward_time_steps."
		assert "optim_time_steps" not in kwargs, f"{self.__class__} does not support optim_time_steps."
		assert params is None, f"{self.__class__} does not support params yet."
		assert layers is not None, f"{self.__class__} requires layers."
		super().__init__(
			params=params,
			layers=layers,
			backward_time_steps=1,
			optim_time_steps=1,
			**kwargs
		)
		self.random_feedbacks = kwargs.get("random_feedbacks", True)
		if not self.random_feedbacks:
			raise NotImplementedError("Non-random feedbacks are not implemented yet.")
		self.feedback_weights = None
		self.rn_gen = torch.Generator()
		self.rn_gen.manual_seed(kwargs.get("seed", 0))
		self.running_grads = None
		self.eligibility_traces = defaultdict(list)
		self.learning_signals = defaultdict(list)
		self.layers_to_params = defaultdict(list)
	
	def load_checkpoint_state(self, trainer, checkpoint: dict, **kwargs):
		if self.save_state:
			state = checkpoint.get(self.name, {})
			opt_state_dict = state.get(self.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY, None)
			if opt_state_dict is not None:
				self.optimizer.load_state_dict(opt_state_dict)
			if self.random_feedbacks:
				self.feedback_weights = state.get(self.CHECKPOINT_RN_FEEDBACK_WEIGHTS_KEY, None)
	
	def get_checkpoint_state(self, trainer, **kwargs) -> object:
		if self.save_state:
			state = {}
			if self.optimizer is not None:
				state[self.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY] = self.optimizer.state_dict()
			if self.random_feedbacks:
				state[self.CHECKPOINT_RN_FEEDBACK_WEIGHTS_KEY] = self.feedback_weights
			return state
		return None
	
	def initialize_running_grads(self):
		self.running_grads = {
			layer.name: [
				torch.zeros_like(p)
				for p in self.layers_to_params[layer.name]
			]
			for layer in self.layers
		}
	
	def initialize_feedback_weights(self, pred_batch: Optional[torch.Tensor] = None):
		if pred_batch is None:
			pred_batch = self.trainer.current_training_state.pred_batch
		self.feedback_weights = {
			layer.name: [
				torch.randn((pred_batch.shape[-1], p.shape[-1]), generator=self.rn_gen)
				for p in self.layers_to_params[layer.name]
			]
			for layer in self.layers
		}
	
	def start(self, trainer, **kwargs):
		LearningAlgorithm.start(self, trainer, **kwargs)
		if self.params and self.optimizer is None:
			self.optimizer = torch.optim.SGD(self.params, lr=self.kwargs.get("lr", 1.0e-3))
		elif not self.params and self.optimizer is not None:
			self.params.extend(
				[
					param
					for i in range(len(self.optimizer.param_groups))
					for param in self.optimizer.param_groups[i]["params"]
				]
			)
		else:
			self.params = list(trainer.model.parameters())
			self.optimizer = torch.optim.SGD(self.params, lr=self.kwargs.get("lr", 1.0e-3))
		
		if self.criterion is None and trainer.criterion is not None:
			self.criterion = trainer.criterion
		
		# filter params to get only the ones that require gradients
		self.params = [param for param in self.params if param.requires_grad]
		assert all(param.ndim == 2 for param in self.params), "Eprop only supports 2D parameters for now."
		self.layers_to_params = {
			layer.name: [
				param for param in layer.parameters() if param.requires_grad
			]
			for layer in self.layers
		}
		self.params = [p for params in self.layers_to_params.values() for p in params]
		self.optimizer = torch.optim.SGD(self.params, lr=self.kwargs.get("lr", 1.0e-3))
		
		self.output_layers: torch.nn.ModuleDict = torch.nn.ModuleDict({layer.name: layer for layer in self.layers})
		self._initialize_original_forwards()
		self.initialize_running_grads()
	
	def on_batch_begin(self, trainer, **kwargs):
		super().on_batch_begin(trainer)
		if trainer.model.training:
			if self.feedback_weights is None:
				self.initialize_feedback_weights(self.trainer.current_training_state.y_batch)
			self.eligibility_traces = defaultdict(list)
			self.learning_signals = defaultdict(list)
			self.initialize_running_grads()
			self._last_et = {
				layer.name: [
					torch.zeros_like(p)
					for p in self.layers_to_params[layer.name]
				]
				for layer in self.layers
			}
			# For Debugging
			self.mean_eligibility_traces = defaultdict(list)
			self.mean_learning_signals = defaultdict(list)
			
	def _get_hidden_tensor(self, out: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]]):
		if isinstance(out, (tuple, list)):
			out = out[-1]
		return out
	
	def decorate_forwards(self):
		if self.trainer.model.training:
			if not self._forwards_decorated:
				self._initialize_original_forwards()
			for layer in self.output_layers.values():
				layer.forward = self._decorate_forward(layer.forward, layer.name)
			self._forwards_decorated = True
	
	def _decorate_forward(self, forward, layer_name: str):
		def _forward(*args, **kwargs):
			out = forward(*args, **kwargs)
			t = kwargs.get("t", None)
			if t is None:
				return out
			out_tensor = self._get_out_tensor(out)
			h_tensor = self._get_hidden_tensor(out)
			if t == 0:  # Hotfix for the first time step  TODO: fix this
				ready = bool(self._layers_buffer[layer_name])
			else:
				ready = True
			list_insert_replace_at(self._layers_buffer[layer_name], t % self.backward_time_steps, out_tensor)
			if len(self._layers_buffer[layer_name]) == self.backward_time_steps and ready:
				self._backward_at_t(t, self.backward_time_steps, layer_name)
			return out
		return _forward
	
	def _backward_at_t(self, t: int, backward_t: int, layer_name: str):
		y_batch = self._get_y_batch_slice_from_trainer((t + 1) - backward_t, t + 1, layer_name)
		# TODO: must have a pred batch for the layer and one for the output_layer to compute the learning signals correctly
		pred_batch = torch.squeeze(self._get_pred_batch_from_buffer(layer_name))
		
		# compute dz/dw
		grad_outputs = torch.eye(pred_batch.shape[-1], device=pred_batch.device)
		params = self.layers_to_params[layer_name]
		instantaneous_eligibility_traces = [torch.zeros_like(p) for p in params]
		# TODO: try to understand what is going on here: https://github.com/IGITUGraz/eligibility_propagation/blob/efd02e6879c01cda3fa9a7838e8e2fd08163c16e/Figure_3_and_S7_e_prop_tutorials/tutorial_pattern_generation.py#L98
		for i in range(grad_outputs.shape[0]):
			zero_grad_params(params)
			
			# pred_batch.backward(grad_outputs[i], retain_graph=True)
			for p_idx, param in enumerate(params):
				# TODO: pas sur du slicing
				instantaneous_eligibility_traces[p_idx][i] = (
					# 0.1 * self._last_et[layer_name][p_idx][i] + param.grad.detach().clone()[i]
					# param.grad.detach().clone()[i]
					torch.autograd.grad(pred_batch[i], param, retain_graph=True)[0][i]
				)
		self._last_et[layer_name] = instantaneous_eligibility_traces
		self.eligibility_traces[layer_name].append(instantaneous_eligibility_traces)
		
		# compute learning signals
		mean_error = torch.mean((y_batch - pred_batch).view(-1, y_batch.shape[-1]), dim=0)
		instantaneous_learning_signals = [
			torch.matmul(mean_error, self.feedback_weights[layer_name][p_idx]).view(1, -1)
			for p_idx in range(len(self.feedback_weights[layer_name]))
		]
		self.learning_signals[layer_name].append(instantaneous_learning_signals)
		self.running_grads[layer_name] = [
			self.running_grads[layer_name][p_idx] + (
					instantaneous_learning_signals[p_idx] * instantaneous_eligibility_traces[p_idx]
			)
			for p_idx in range(len(self.running_grads[layer_name]))
		]
		self._layers_buffer[layer_name].clear()
		
		# For Debugging
		self.mean_eligibility_traces[layer_name].append(
			to_numpy(torch.mean(torch.abs(torch.cat(instantaneous_eligibility_traces, dim=0)))).item()
		)
		self.mean_learning_signals[layer_name].append(
			to_numpy(torch.mean(torch.abs(torch.cat(instantaneous_learning_signals, dim=0)))).item()
		)
	
	def apply_grads(self):
		for layer_name, params in self.layers_to_params.items():
			for p_idx, param in enumerate(params):
				param.grad = self.running_grads[layer_name][p_idx].detach().clone()
	
	def on_optimization_begin(self, trainer, **kwargs):
		y_batch = trainer.current_training_state.y_batch
		pred_batch = trainer.format_pred_batch(trainer.current_training_state.pred_batch, y_batch)
		self.optimizer.zero_grad()
		self.apply_grads()
		self.optimizer.step()
		trainer.update_state_(batch_loss=self.apply_criterion(pred_batch, y_batch).detach_())
	
	def on_optimization_end(self, trainer, **kwargs):
		self.optimizer.zero_grad()
	
	def on_pbar_update(self, trainer, **kwargs) -> dict:
		return {
			# "mean_grads": {
			# 	layer_name: [
			# 		to_numpy(torch.mean(torch.abs(p.grad.detach().clone()))).item()
			# 		for p in self.running_grads[layer_name]
			# 	]
			# 	for layer_name in self.running_grads
			# },
			# "mean_eligibility_traces": self.mean_eligibility_traces,
			# "mean_learning_signals": self.mean_learning_signals,
		}
	
