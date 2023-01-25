import warnings
from collections import defaultdict
from typing import Optional, Sequence, Union, Dict, Callable, Tuple, List, Mapping

import torch
from unstable import unstable

from .learning_algorithm import LearningAlgorithm
from ..transforms.base import to_numpy
from ..learning_algorithms.tbptt import TBPTT
from ..utils import (
	batchwise_temporal_filter,
	list_insert_replace_at,
	zero_grad_params,
	unpack_out_hh,
	recursive_detach,
	filter_parameters,
	dy_dw_local
)


# @unstable
class Eprop(TBPTT):
	r"""
	Apply the eligibility trace forward propagation (e-prop) :cite:t:`bellec_solution_2020`
	algorithm to the given model.
	"""
	CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: str = "optimizer_state_dict"
	CHECKPOINT_FEEDBACK_WEIGHTS_KEY: str = "feedback_weights"
	OPTIMIZER_PARAMS_GROUP_IDX = 0
	OPTIMIZER_OUTPUT_PARAMS_GROUP_IDX = 1
	DEFAULT_OPTIMIZER_CLS = torch.optim.Adam
	DEFAULT_Y_KEY = "default_key"
	
	def __init__(
			self,
			*,
			params: Optional[Sequence[torch.nn.Parameter]] = None,
			output_params: Optional[Sequence[torch.nn.Parameter]] = None,
			layers: Optional[Union[Sequence[torch.nn.Module], torch.nn.Module]] = None,
			output_layers: Optional[Union[Sequence[torch.nn.Module], torch.nn.Module]] = None,
			**kwargs
	):
		"""
		Constructor for Eprop class.

		:param params: The parameters to optimize. If None, the parameters of the model's trainer will be used.
		:type params: Optional[Sequence[torch.nn.Parameter]]
		:param optimizer: The optimizer to use. If provided make sure to provide the param_group in the following format:
								[{"params": params, "lr": params_lr}, {"params": output_params, "lr": output_params_lr}]
						The index of the group must be the same as the OPTIMIZER_PARAMS_GROUP_IDX and
						OPTIMIZER_OUTPUT_PARAMS_GROUP_IDX constants which are 0 and 1 respectively.
						If not provided, torch.optim.Adam is used.
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
		kwargs.setdefault("backward_time_steps", 1)
		kwargs.setdefault("optim_time_steps", 1)
		# TODO: implement a optim step at each `optim_time_steps` steps.
		# assert "backward_time_steps" not in kwargs, f"{self.__class__} does not support backward_time_steps."
		# assert "optim_time_steps" not in kwargs, f"{self.__class__} does not support optim_time_steps."
		# assert params is None, f"{self.__class__} does not support params yet."
		# assert layers is not None, f"{self.__class__} requires layers."
		super().__init__(
			params=params,
			layers=layers,
			**kwargs
		)
		self.output_params = output_params
		self.output_layers = output_layers
		self.random_feedbacks = kwargs.get("random_feedbacks", True)
		if not self.random_feedbacks:
			raise NotImplementedError("Non-random feedbacks are not implemented yet.")
		self.feedback_weights = None
		self.rn_gen = torch.Generator()
		self.rn_gen.manual_seed(kwargs.get("seed", 0))
		self.eligibility_traces = None
		self.learning_signals = defaultdict(list)
		self.param_groups = []
		self._hidden_layer_names = []
		self.eval_criterion = kwargs.get("eval_criterion", self.criterion)
	
	def load_checkpoint_state(self, trainer, checkpoint: dict, **kwargs):
		if self.save_state:
			state = checkpoint.get(self.name, {})
			opt_state_dict = state.get(self.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY, None)
			if opt_state_dict is not None:
				saved_param_groups = opt_state_dict["param_groups"]
				if self.optimizer is None:
					self.optimizer = self.DEFAULT_OPTIMIZER_CLS(saved_param_groups)
				self.optimizer.load_state_dict(opt_state_dict)
				self.param_groups = self.optimizer.param_groups
				self.params = self.param_groups[self.OPTIMIZER_PARAMS_GROUP_IDX]["params"]
				self.output_params = self.param_groups[self.OPTIMIZER_OUTPUT_PARAMS_GROUP_IDX]["params"]
			if self.random_feedbacks:
				self.feedback_weights = state.get(self.CHECKPOINT_FEEDBACK_WEIGHTS_KEY, None)
	
	def get_checkpoint_state(self, trainer, **kwargs) -> object:
		if self.save_state:
			state = {}
			if self.optimizer is not None:
				state[self.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY] = self.optimizer.state_dict()
			if self.random_feedbacks:
				state[self.CHECKPOINT_FEEDBACK_WEIGHTS_KEY] = self.feedback_weights
			return state
		return None
	
	def initialize_feedback_weights(self, y_batch: Optional[Union[Dict[str, torch.Tensor], torch.Tensor]] = None):
		"""
		TODO : Possibility to initialize the feedback weights with any initialization methods
		TODO : Non-random feedbacks must be implemented with {W_out}.T
		Initialize the feedback weights for each params.
		The random feedback is noted B_{ij} in Bellec's paper :cite:t:`bellec_solution_2020`
		.

		:param y_batch:
		:return:
		"""
		if y_batch is None:
			y_batch = self.trainer.current_training_state.y_batch
		if not isinstance(y_batch, dict):
			y_batch = {self.DEFAULT_Y_KEY: y_batch}
		last_dims = [p.shape[-1] if p.ndim > 0 else 1 for p in self.params]
		if self.random_feedbacks:
			self.feedback_weights = {
				k: [
					torch.randn((y_batch_item.shape[-1], pld), generator=self.rn_gen)
					for pld in last_dims
				] for k, y_batch_item in y_batch.items()
			}
		else:
			raise NotImplementedError("Non-random feedbacks are not implemented yet.")
	
	def _initialize_original_forwards(self):
		for layer in self.trainer.model.get_all_layers():
			self._original_forwards[layer.name] = (layer, layer.forward)
	
	def initialize_params(self, trainer=None):
		"""
		Initialize the parameters of the optimizer.

		:Note: Must be called after :meth:`initialize_output_params` and :meth:`initialize_layers`.
		
		:param trainer: The trainer to use.
		:return: None
		"""
		if not self.params and self.optimizer:
			self.params = self.optimizer.param_groups[self.OPTIMIZER_PARAMS_GROUP_IDX]["params"]

		if not self.params:
			self.params = [
				param
				for layer in self.layers
				for param in layer.parameters()
			]
		if not self.params:
			warnings.warn("No hidden parameters found. Please provide them manually if you have any.")
		
		# self.params = [p for p in self.params if p not in self.output_params]
		# self.params = filter_parameters(self.params, requires_grad=True)
		return
	
	def initialize_output_params(self, trainer=None):
		"""
		Initialize the output parameters of the optimizer. Try multiple ways to identify the
		output parameters if those are not provided by the user.

		:Note: Must be called after :meth:`initialize_output_layers`.

		:param trainer: The trainer object
		:return: None
		"""
		if not self.output_params and self.optimizer:
			self.output_params = self.optimizer.param_groups[self.OPTIMIZER_OUTPUT_PARAMS_GROUP_IDX]["params"]

		if not self.output_params:
			self.output_params = [
				param
				for layer in self.output_layers
				for param in layer.parameters()
			]
		if not self.output_params:
			raise ValueError("Could not find output parameters. Please provide them manually.")
		
		# self.output_params = filter_parameters(self.output_params, requires_grad=True)
		# self.params = [p for p in self.params if p not in self.output_params]
		return

	def initialize_output_layers(self, trainer):
		"""
		Initialize the output layers of the optimizer. Try multiple ways to identify the output layers if those are not
		provided by the user.

		:Note: Must be called before :meth:`initialize_output_params`.

		:param trainer: The trainer object.
		:return: None
		"""
		if not self.output_layers:
			self.output_layers = []
			possible_attrs = ["output_layers", "output_layer"]
			for attr in possible_attrs:
				obj = getattr(trainer.model, attr, [])
				if isinstance(obj, (Sequence, torch.nn.ModuleList)):
					obj = list(obj)
				elif isinstance(obj, (Mapping, torch.nn.ModuleDict)):
					obj = list(obj.values())
				elif isinstance(obj, torch.nn.Module):
					obj = [obj]
				self.output_layers += list(obj)

		if not self.output_layers:
			raise ValueError(
				"Could not find output layers. Please provide them manually."
			)

	def initialize_layers(self, trainer):
		"""
		Initialize the layers of the optimizer. Try multiple ways to identify the output layers if those are not
		provided by the user.

		:param trainer:
		:return:
		"""
		if not self.layers:
			self.layers = []
			possible_attrs = ["input_layers", "input_layer", "hidden_layers", "hidden_layer"]
			for attr in possible_attrs:
				if hasattr(trainer.model, attr):
					obj = getattr(trainer.model, attr, [])
					if isinstance(obj, (Sequence, torch.nn.ModuleList)):
						obj = list(obj)
					elif isinstance(obj, (Mapping, torch.nn.ModuleDict)):
						obj = list(obj.values())
					elif isinstance(obj, torch.nn.Module):
						obj = [obj]
					self.layers += list(obj)
		if not self.layers:
			warnings.warn(
				"No hidden layers found. Please provide them manually if you have any."
				"If you are using only one layer, please note that E-prop is equivalent to a TBPTT. If this is the"
				"case, one might consider using TBPTT instead of E-prop."
			)

	def initialize_param_groups(self):
		"""
		The learning rate are initialize. If the user has provided a learning rate for each parameter, then it is used.

		:return:
		"""
		self.param_groups = []
		list_insert_replace_at(
			self.param_groups,
			self.OPTIMIZER_PARAMS_GROUP_IDX,
			{"params": self.params, "lr": self.kwargs.get("params_lr", 1e-4)}
		)
		list_insert_replace_at(
			self.param_groups,
			self.OPTIMIZER_OUTPUT_PARAMS_GROUP_IDX,
			{"params": self.output_params, "lr": self.kwargs.get("output_params_lr", 2e-4)}
		)

	def start(self, trainer, **kwargs):
		"""
		Start the training process with E-prop.

		:param trainer:
		:param kwargs:
		:return:
		"""
		LearningAlgorithm.start(self, trainer, **kwargs)
		self.initialize_output_layers(trainer)
		self.initialize_output_params(trainer)
		self.initialize_layers(trainer)
		self.initialize_params(trainer)

		if self.criterion is None and trainer.criterion is not None:
			self.criterion = trainer.criterion
		
		if not self.param_groups:
			self.initialize_param_groups()
		if not self.optimizer:
			self.optimizer = self.DEFAULT_OPTIMIZER_CLS(self.param_groups)
		
		self._initialize_original_forwards()
	
	def on_batch_begin(self, trainer, **kwargs):
		"""
		For each batch. Initialize the random feedback weights if not already done. Also, set the eligibility traces
		to zero.

		:param trainer:
		:param kwargs:
		:return:
		"""
		super().on_batch_begin(trainer)
		if trainer.model.training:
			if self.feedback_weights is None:
				self.initialize_feedback_weights(trainer.current_training_state.y_batch)
		self.eligibility_traces = [torch.zeros_like(p) for p in self.params]
		zero_grad_params(self.params)
		zero_grad_params(self.output_params)
	
	def decorate_forwards(self):
		"""
		Ensure that the forward pass is decorated. THe original forward and the hidden layers names are stored. The
		hidden layers forward method are decorated using :meth: `_decorate_hidden_forward`. The output layers forward
		are decorated using :meth: `_decorate_output_forward` from TBPTT.

		Here, we are using decorators to introduce a specific behavior in the forward pass. For E-prop, we need to
		ensure that the gradient is computed and optimize at each time step t of the sequence. This can be achieved by
		decorating our forward. However, we do keep in storage the previous forward pass. This is done to ensure
		that the forward pass is not modified permanently in any way.

		"""
		if self.trainer.model.training:
			if not self._forwards_decorated:
				self._initialize_original_forwards()
			self._hidden_layer_names.clear()
			
			for layer in self.layers:
				layer.forward = self._decorate_hidden_forward(layer.forward, layer.name)
				self._hidden_layer_names.append(layer.name)
			
			for layer in self.output_layers:
				layer.forward = self._decorate_forward(layer.forward, layer.name)
			self._forwards_decorated = True
	
	def _decorate_hidden_forward(self, forward, layer_name: str):
		"""
		In TBPTT, we decorate forward to ensure that the backpropagation and the optimizer at t is done for the entire
		network. In E-prop, we want to backpropagate the hidden layers locally (and not the entire network) at each
		time step t.

		:param forward:
		:param layer_name:
		:return:
		"""
		def _forward(*args, **kwargs):
			out = forward(*args, **kwargs)
			t = kwargs.get("t", None)
			if t is None:
				return out
			out_tensor, hh = unpack_out_hh(out)
			if t == 0:  # Hotfix for the first time step  TODO: fix this
				ready = bool(self._layers_buffer[layer_name])
			else:
				ready = True
			list_insert_replace_at(self._layers_buffer[layer_name], t % self.backward_time_steps, out_tensor)
			length = len(self._layers_buffer[layer_name])
			if length == self.backward_time_steps and ready:
				self._hidden_backward_at_t(t, self.backward_time_steps, layer_name)
				out = recursive_detach(out)
			return out
		return _forward
	
	def _hidden_backward_at_t(self, t: int, backward_t: int, layer_name: str):
		"""
		TODO : Filter with kappa must be added in order to train SNNs
		Here, we compute the eligibility trace as seen in equation (13) from :cite:t:`bellec_solution_2020`. Please
		note that while the notation used in this paper for the equation (13) is [dz/dW]_{local}, we have used [dy/dW]
		in order to be coherent with our own convention.

		:param t:
		:param backward_t:
		:param layer_name:
		:return:
		"""
		pred_batch = torch.squeeze(self._get_pred_batch_from_buffer(layer_name))
		dy_dw_locals = dy_dw_local(y=pred_batch, params=self.params, retain_graph=True, allow_unused=True)
		self.eligibility_traces = [et+dy_dw for et, dy_dw in zip(self.eligibility_traces, dy_dw_locals)]
		self._layers_buffer[layer_name].clear()
	
	def _backward_at_t(self, t: int, backward_t: int, layer_name: str):
		"""
		Apply the criterion on the batch. The gradients of each parameters are then updated but are not yet optimized.

		:param t:
		:param backward_t:
		:param layer_name:
		:return:
		"""
		y_batch = self._get_y_batch_slice_from_trainer((t + 1) - backward_t, t + 1, layer_name)
		pred_batch = self._get_pred_batch_from_buffer(layer_name)
		batch_loss = self.apply_criterion(pred_batch, y_batch)
		if batch_loss.grad_fn is None:
			raise ValueError(
				f"batch_loss.grad_fn is None. This is probably an internal error. Please report this issue on GitHub."
			)
		errors = self.compute_errors(pred_batch, y_batch)
		self.update_grads(errors, batch_loss)
		self._layers_buffer[layer_name].clear()
		
	def compute_learning_signals(self, errors: Dict[str, torch.Tensor]):
		"""
		TODO : Determine if we normalize with the number of output when computing the learning signal.
		The learning signals are computed using equation (28) from :cite:t:`bellec_solution_2020`.

		:param errors:
		:return:
		"""
		learning_signals = [torch.zeros((p.shape[0] if p.ndim > 0 else 1), device=p.device) for p in self.params]
		for k, feedbacks in self.feedback_weights.items():
			if k not in errors:
				raise ValueError(
					f"This is an internal error. Please report this issue on GitHub."
					f"Key {k} from {self.feedback_weights.keys()=} not found in errors of keys {errors.keys()}."
				)
			error_mean = torch.mean(errors[k].view(-1, errors[k].shape[-1]), dim=0)
			for i, feedback in enumerate(feedbacks):
				learning_signals[i] = learning_signals[i] + torch.matmul(error_mean, feedback.to(error_mean.device))
		return learning_signals
	
	def compute_errors(
			self,
			pred_batch: Union[Dict[str, torch.Tensor], torch.Tensor],
			y_batch: Union[Dict[str, torch.Tensor], torch.Tensor]
	) -> Dict[str, torch.Tensor]:
		"""
		The errors for each output is computed then inserted in a dict for further use. This function check if the
		y_batch and pred_batch are given as a dict or a tensor.

		:param pred_batch:
		:param y_batch:
		:return:
		"""
		if isinstance(y_batch, dict) or isinstance(pred_batch, dict):
			if isinstance(y_batch, torch.Tensor):
				y_batch = {k: y_batch for k in pred_batch}
			else:
				raise ValueError(f"y_batch must be a dict or a tensor, not {type(y_batch)}.")
			if isinstance(pred_batch, torch.Tensor):
				pred_batch = {k: pred_batch for k in y_batch}
			else:
				raise ValueError(f"pred_batch must be a dict or a tensor, not {type(pred_batch)}.")
			batch_errors = {
				k: (pred_batch[k] - y_batch[k].to(pred_batch[k].device))
				for k in y_batch
			}
		else:
			batch_errors = {self.DEFAULT_Y_KEY: pred_batch - y_batch.to(pred_batch.device)}
		return batch_errors
	
	def update_grads(
			self,
			error: Dict[str, torch.Tensor],
			loss: torch.Tensor,
	):
		"""
		The learning signal is computed. The gradients of the parameters are then updated as seen in equation (28)
		from :cite:t:`bellec_solution_2020`.

		:param error:
		:param loss:
		:return:
		"""
		learning_signals = self.compute_learning_signals(error)
		with torch.no_grad():
			for param, ls, et in zip(self.params, learning_signals, self.eligibility_traces):
				if param.requires_grad:
					param.grad += (ls * et.to(ls.device)).to(param.device).view(param.shape).detach()

		output_grads = dy_dw_local(torch.mean(loss), self.output_params, retain_graph=True, allow_unused=True)
		with torch.no_grad():
			for out_param, out_grad in zip(self.output_params, output_grads):
				if out_param.requires_grad:
					out_param.grad += out_grad
	
	def _make_optim_step(self, **kwargs):
		"""
		TODO: check if we really need to zero_grad -> backward_times_steps is not necessary equal to optim_times_steps.
		Set the gradients and the eligibility traces to zero.

		:param kwargs:
		:return:
		"""
		super()._make_optim_step(**kwargs)
		zero_grad_params(self.output_params)
		self.eligibility_traces = [torch.zeros_like(p) for p in self.params]
	
	def on_batch_end(self, trainer, **kwargs):
		"""
		Ensure that there is not any remaining gradients in the output parameters. The forward are undecorated and the
		gradients are set to zero. The buffer are also cleared.

		:param trainer:
		:param kwargs:
		:return:
		"""
		super().on_batch_end(trainer)
		if trainer.model.training:
			need_optim_step = False
			for layer_name in self._layers_buffer:
				backward_t = len(self._layers_buffer[layer_name])
				if backward_t > 0:
					need_optim_step = True
					if layer_name in self._hidden_layer_names:
						self._hidden_backward_at_t(self._data_n_time_steps - 1, backward_t, layer_name)
					else:
						self._backward_at_t(self._data_n_time_steps - 1, backward_t, layer_name)
			if need_optim_step:
				self._make_optim_step()
		self.undecorate_forwards()
		self._layers_buffer.clear()
		self.optimizer.zero_grad()

	
