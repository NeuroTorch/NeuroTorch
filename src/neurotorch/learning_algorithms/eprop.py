import warnings
from collections import defaultdict
from typing import Optional, Sequence, Union, Dict, Callable

import torch

from .learning_algorithm import LearningAlgorithm
from ..learning_algorithms.tbptt import TBPTT
from ..utils import batchwise_temporal_filter, list_insert_replace_at, zero_grad_params


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
		warnings.warn("Eprop is still in beta and may not work as expected or act exactly as BPTT.")
		kwargs.setdefault("save_state", True)
		kwargs.setdefault("load_state", True)
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
		self.rn_feedback_weights = None
		self.rn_gen = torch.Generator()
		self.rn_gen.manual_seed(kwargs.get("seed", 0))
		self.eligibility_traces = None
		self.layers_to_params = defaultdict(list)
	
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
	
	def initialize_eligibility_traces(self):
		self.eligibility_traces = {
			layer.name: [
				torch.zeros_like(p)
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
		
		self.output_layers: torch.nn.ModuleDict = torch.nn.ModuleDict({layer.name: layer for layer in self.layers})
		self._initialize_original_forwards()
		self.initialize_eligibility_traces()
		
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
		pred_batch = torch.squeeze(self._get_pred_batch_from_buffer(layer_name))
		grad_outputs = torch.eye(pred_batch.shape[-1], device=pred_batch.device)
		params = self.layers_to_params[layer_name]
		for i in range(grad_outputs.shape[0]):
			zero_grad_params(params)
			pred_batch.backward(grad_outputs[i], retain_graph=True)
			for p_idx, param in enumerate(params):
				self.eligibility_traces[layer_name][p_idx][i] += param.grad.detach().clone()[i]
		self._layers_buffer[layer_name].clear()
	
	def get_learning_signals(self, trainer, **kwargs):
		# TODO: faire pour seulement un pas de temps
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
		# TODO: peut-être qu'il faudrait mette un décorateur comme dans tbptt
		# TODO: le psi de l'équation (23) de l'article de e-prop est essentiellement la dérivé de la fonction d'activation
		# TODO: de la layer. Il faut donc que je trouve un moyen de récupérer cette dérivé.
		y_batch = trainer.current_training_state.y_batch
		pred_batch = trainer.format_pred_batch(trainer.current_training_state.pred_batch, y_batch)
		eligibility_vectors = batchwise_temporal_filter(pred_batch)
		return eligibility_vectors
	
	def apply_grads(self, learning_signals, eligibility_traces):
		for i, (w, L, epsilon) in enumerate(zip(self.params, learning_signals, eligibility_traces)):
			grad = torch.matmul(epsilon, L.T)
			w.grad = grad
	
	def on_optimization_begin(self, trainer, **kwargs):
		self.optimizer.zero_grad()
		learning_signals = self.get_learning_signals(trainer, **kwargs)
		eligibility_traces = self.get_eligibility_trace(trainer, **kwargs)
		self.apply_grads(learning_signals, eligibility_traces)
		self.optimizer.step()
	
	def on_optimization_end(self, trainer, **kwargs):
		self.optimizer.zero_grad()
	
