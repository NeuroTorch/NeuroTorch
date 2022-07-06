import enum
from copy import deepcopy
from typing import Optional, Sized, Tuple, Type, Union, Iterable

import numpy as np
import torch
from torch import nn

from . import HeavisideSigmoidApprox, SpikeFunction
from ..dimension import Dimension, DimensionProperty, DimensionsLike, SizeTypes


class LearningType(enum.Enum):
	NONE = 0
	BACKPROP = 1
	E_PROP = 2


class LayerType(enum.Enum):
	LIF = 0
	ALIF = 1
	Izhikevich = 2
	LI = 3


class BaseLayer(torch.nn.Module):
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			learning_type: LearningType = LearningType.BACKPROP,
			device: Optional[torch.device] = None,
			**kwargs
	):
		super(BaseLayer, self).__init__()
		self._is_built = False
		self._name_is_set = False
		self.name = name

		self.learning_type = learning_type
		self.device = device
		if self.device is None:
			self._set_default_device_()

		self.kwargs = kwargs
		self._set_default_kwargs()

		self.input_size = input_size
		self.output_size = output_size

	@property
	def input_size(self):
		if not hasattr(self, "_input_size"):
			return None
		return self._input_size

	@input_size.setter
	def input_size(self, size: Optional[SizeTypes]):
		self._input_size = self._format_size(size)

	@property
	def output_size(self):
		if not hasattr(self, "_output_size"):
			return None
		return self._output_size

	@output_size.setter
	def output_size(self, size: Optional[SizeTypes]):
		self._output_size = self._format_size(size)
	
	@property
	def requires_grad(self):
		return self.learning_type == LearningType.BACKPROP

	@property
	def name(self) -> str:
		if self._name is None:
			return self.__class__.__name__
		return self._name

	@property
	def name_is_set(self) -> bool:
		return self._name_is_set

	@name.setter
	def name(self, name: str):
		self._name = name
		if name is not None:
			assert isinstance(name, str), "name must be a string."
			self._name_is_set = True

	@property
	def is_ready_to_build(self) -> bool:
		return all([
			s is not None
			for s in [
				self._input_size,
				(self._output_size if hasattr(self, "_output_size") else None)
			]
		])

	@property
	def is_built(self) -> bool:
		return self._is_built

	def _format_size(self, size: Optional[SizeTypes]) -> Optional[DimensionsLike]:
		# TODO: must accept multiple time dimensions
		if size is not None:
			if isinstance(size, Iterable):
				size = [Dimension.from_int_or_dimension(s) for s in size]
				time_dim_count = len(list(filter(lambda d: d.dtype == DimensionProperty.TIME, size)))
				assert time_dim_count <= 1, "Size must not contain more than one Time dimension."
				size = list(filter(lambda d: d.dtype != DimensionProperty.TIME, size))
				if len(size) == 1:
					size = size[0]
				else:
					raise ValueError("Size must be a single dimension or a list of 2 dimensions with a Time one.")
			assert isinstance(size, (int, Dimension)), "Size must be an int or Dimension."
			size = Dimension.from_int_or_dimension(size)
		return size

	def _set_default_kwargs(self):
		pass

	def _set_default_device_(self):
		self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	def build(self):
		self._is_built = True

	def create_empty_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
		raise NotImplementedError()

	def _init_forward_state(
			self,
			state: Tuple[torch.Tensor, ...] = None,
			batch_size: int = 1
	) -> Tuple[torch.Tensor, ...]:
		if state is None:
			state = self.create_empty_state(batch_size)
		elif any([e is None for e in state]):
			empty_state = self.create_empty_state(batch_size)
			state = list(state)
			for i, e in enumerate(state):
				if e is None:
					state[i] = empty_state[i]
			state = tuple(state)
		return state

	def infer_sizes_from_inputs(self, inputs: torch.Tensor):
		self.input_size = inputs.shape[-1]
		if self.output_size is None:
			raise ValueError("output_size must be specified before the forward call.")
	
	def __call__(self, inputs: torch.Tensor, *args, **kwargs):
		if not self.is_built:
			self.infer_sizes_from_inputs(inputs)
			self.build()
		return super(BaseLayer, self).__call__(inputs, *args, **kwargs)

	def forward(self, inputs: torch.Tensor, state: torch.Tensor = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
		raise NotImplementedError

	def initialize_weights_(self):
		for param in self.parameters():
			if param.ndim > 2:
				torch.nn.init.xavier_normal_(param)
			else:
				torch.nn.init.normal_(param)


class BaseNeuronsLayer(BaseLayer):
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			use_recurrent_connection: bool = True,
			use_rec_eye_mask: bool = True,
			learning_type: LearningType = LearningType.BACKPROP,
			dt: float = 1e-3,
			device: Optional[torch.device] = None,
			**kwargs
	):
		self.dt = dt
		super().__init__(
			input_size=input_size,
			output_size=output_size,
			name=name,
			learning_type=learning_type,
			device=device,
			**kwargs
		)
		self.use_recurrent_connection = use_recurrent_connection
		self.forward_weights = None
		self.use_rec_eye_mask = use_rec_eye_mask
		self.recurrent_weights = None
		self.rec_mask = None

	def create_empty_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
		raise NotImplementedError()

	def forward(self, inputs: torch.Tensor, state: torch.Tensor = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
		raise NotImplementedError()

	def build(self):
		super().build()
		self.forward_weights = nn.Parameter(
			torch.empty((int(self.input_size), int(self.output_size)), device=self.device, dtype=torch.float32),
			requires_grad=self.requires_grad
		)
		if self.use_recurrent_connection:
			self.recurrent_weights = nn.Parameter(
				torch.empty((int(self.output_size), int(self.output_size)), device=self.device, dtype=torch.float32),
				requires_grad=self.requires_grad
			)
			if self.use_rec_eye_mask:
				self.rec_mask = (1 - torch.eye(int(self.output_size), device=self.device, dtype=torch.float32))
			else:
				self.rec_mask = torch.ones(
					(int(self.output_size), int(self.output_size)), device=self.device, dtype=torch.float32
				)
		self.initialize_weights_()


class LIFLayer(BaseNeuronsLayer):
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			use_recurrent_connection=True,
			use_rec_eye_mask=True,
			spike_func: Type[SpikeFunction] = HeavisideSigmoidApprox,
			learning_type: LearningType = LearningType.BACKPROP,
			dt=1e-3,
			device=None,
			**kwargs
	):
		self.spike_func = spike_func
		super(LIFLayer, self).__init__(
			input_size=input_size,
			output_size=output_size,
			name=name,
			use_recurrent_connection=use_recurrent_connection,
			use_rec_eye_mask=use_rec_eye_mask,
			learning_type=learning_type,
			dt=dt,
			device=device,
			**kwargs
			)

		self.alpha = torch.tensor(np.exp(-dt / self.kwargs["tau_m"]), dtype=torch.float32, device=self.device)
		self.threshold = torch.tensor(self.kwargs["threshold"], dtype=torch.float32, device=self.device)
		self.gamma = torch.tensor(self.kwargs["gamma"], dtype=torch.float32, device=self.device)
		self._regularization_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

	def _set_default_kwargs(self):
		self.kwargs.setdefault("tau_m", 10.0 * self.dt)
		self.kwargs.setdefault("threshold", 1.0)
		if isinstance(self.spike_func, HeavisideSigmoidApprox):
			self.kwargs.setdefault("gamma", 100.0)
		else:
			self.kwargs.setdefault("gamma", 1.0)

	def initialize_weights_(self):
		gain = self.threshold.data
		for param in self.parameters():
			if param.ndim > 2:
				torch.nn.init.xavier_normal_(param, gain=gain)
			else:
				torch.nn.init.normal_(param, std=gain)

	def create_empty_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
		"""
		Create an empty state in the following form:
			([membrane potential of shape (batch_size, self.output_size)],
			[spikes of shape (batch_size, self.output_size)])
		:param batch_size: The size of the current batch.
		:return: The current state.
		"""
		state = tuple([torch.zeros(
			(batch_size, int(self._output_size)),
			device=self.device,
			dtype=torch.float32,
			requires_grad=True,
		) for _ in range(2)])
		return state

	def get_regularization_loss(self):
		"""
		Get and reset the regularization loss for this layer. The regularization loss will be zeroed after it is
		returned.
		:return: The regularization loss.
		"""
		loss = self._regularization_loss
		self._regularization_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
		return loss

	def _update_regularization_loss(self, state):
		"""
		Update the regularization loss for this layer. Each update call increments the regularization loss so at the end
		the regularization loss will be the sum of all calls to this function.
		:param state: The current state of the layer.
		:return: The updated regularization loss.
		"""
		next_V, next_Z = state
		self._regularization_loss += 2e-6*torch.sum(next_Z)
		# self._regularization_loss += 2e-6*torch.mean(torch.sum(next_Z, dim=-1)**2)
		return self._regularization_loss

	def forward(self, inputs: torch.Tensor, state: Tuple[torch.Tensor, ...] = None):
		assert inputs.ndim == 2
		batch_size, nb_features = inputs.shape
		V, Z = self._init_forward_state(state, batch_size)
		input_current = torch.matmul(inputs, self.forward_weights)
		if self.use_recurrent_connection:
			rec_current = torch.matmul(Z, torch.mul(self.recurrent_weights, self.rec_mask))
		else:
			rec_current = 0.0
		next_V = (self.alpha * V + input_current + rec_current) * (1.0 - Z.detach())
		next_Z = self.spike_func.apply(next_V, self.threshold, self.gamma)

		self._update_regularization_loss((next_V, next_Z))
		return next_Z, (next_V, next_Z)


class SpyLIFLayer(BaseNeuronsLayer):
	"""
	The SpyLIF layer is a LIF layer implemented by the SpyTorch library
	(https://github.com/surrogate-gradient-learning/spytorch) from the
	paper: https://ieeexplore.ieee.org/document/8891809. In this LIF varient they are using a second differential
	equation to model the synaptic current. The second hidden state of the layer add more memory to the model since
	it is not reset every spike like the membrane potential.
	"""
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			use_recurrent_connection: bool = True,
			use_rec_eye_mask: bool = False,
			spike_func: Type[SpikeFunction] = HeavisideSigmoidApprox,
			learning_type: LearningType = LearningType.BACKPROP,
			dt: float = 1e-3,
			device: Optional[torch.device] = None,
			**kwargs
	):
		"""
		Constructor for the SpyLIF layer.
		:param input_size:
		:param output_size:
		:param name:
		:param use_recurrent_connection:
		:param use_rec_eye_mask:
		:param spike_func:
		:param learning_type:
		:param dt:
		:param device:
		:param kwargs:
		"""
		self.spike_func = spike_func
		super(SpyLIFLayer, self).__init__(
			input_size=input_size,
			output_size=output_size,
			name=name,
			use_recurrent_connection=use_recurrent_connection,
			use_rec_eye_mask=use_rec_eye_mask,
			learning_type=learning_type,
			dt=dt,
			device=device,
			**kwargs
			)

		self.alpha = torch.tensor(np.exp(-dt / self.kwargs["tau_syn"]), dtype=torch.float32, device=self.device)
		self.beta = torch.tensor(np.exp(-dt / self.kwargs["tau_mem"]), dtype=torch.float32, device=self.device)
		self.threshold = torch.tensor(self.kwargs["threshold"], dtype=torch.float32, device=self.device)
		self.gamma = torch.tensor(self.kwargs["gamma"], dtype=torch.float32, device=self.device)
		self._regularization_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
		self._regularization_l1 = torch.tensor(0.0, dtype=torch.float32, device=self.device)
		self._n_spike_per_neuron = torch.zeros(int(self.output_size), dtype=torch.float32, device=self.device)
		self._total_count = 0

	def _set_default_kwargs(self):
		self.kwargs.setdefault("tau_syn", 5.0 * self.dt)
		self.kwargs.setdefault("tau_mem", 10.0 * self.dt)
		self.kwargs.setdefault("threshold", 1.0)
		if isinstance(self.spike_func, HeavisideSigmoidApprox):
			self.kwargs.setdefault("gamma", 100.0)
		else:
			self.kwargs.setdefault("gamma", 1.0)

	def initialize_weights_(self):
		weight_scale = 0.2
		torch.nn.init.normal_(self.forward_weights, mean=0.0, std=weight_scale/np.sqrt(int(self.input_size)))
		if self.use_recurrent_connection:
			torch.nn.init.normal_(self.recurrent_weights, mean=0.0, std=weight_scale/np.sqrt(int(self.output_size)))

	def create_empty_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
		"""
		Create an empty state in the following form:
			([membrane potential of shape (batch_size, self.output_size)],
			[synaptic current of shape (batch_size, self.output_size)],
			[spikes of shape (batch_size, self.output_size)])
		:param batch_size: The size of the current batch.
		:return: The current state.
		"""
		state = tuple([torch.zeros(
			(batch_size, int(self._output_size)),
			device=self.device,
			dtype=torch.float32,
			requires_grad=True,
		) for _ in range(3)])
		return state

	def get_regularization_loss(self):
		"""
		Get and reset the regularization loss for this layer. The regularization loss will be zeroed after it is
		returned.
		:return: The regularization loss.
		"""
		loss = self._regularization_loss
		self._regularization_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
		self._regularization_l1 = torch.tensor(0.0, dtype=torch.float32, device=self.device)
		self._n_spike_per_neuron = torch.zeros(int(self.output_size), dtype=torch.float32, device=self.device)
		self._total_count = 0
		return loss

	def _update_regularization_loss(self, state):
		"""
		Update the regularization loss for this layer. Each update call increments the regularization loss so at the end
		the regularization loss will be the sum of all calls to this function.
		:param state: The current state of the layer.
		:return: The updated regularization loss.
		"""
		next_V, next_I_syn, next_Z = state
		self._regularization_l1 += 2e-6*torch.sum(next_Z)
		# self._n_spike_per_neuron += torch.sum(torch.sum(next_Z, dim=0), dim=0)
		# self._total_count += next_Z.shape[0]*next_Z.shape[1]
		# current_l2 = 2e-6*torch.sum(self._n_spike_per_neuron ** 2) / (self._total_count + 1e-6)
		# self._regularization_loss = self._regularization_l1 + current_l2
		self._regularization_loss = self._regularization_l1
		return self._regularization_loss

	def forward(self, inputs: torch.Tensor, state: Tuple[torch.Tensor, ...] = None):
		assert inputs.ndim == 2
		batch_size, nb_features = inputs.shape
		V, I_syn, Z = self._init_forward_state(state, batch_size)
		input_current = torch.matmul(inputs, self.forward_weights)
		if self.use_recurrent_connection:
			rec_current = torch.matmul(Z, torch.mul(self.recurrent_weights, self.rec_mask))
		else:
			rec_current = 0.0
		next_I_syn = self.alpha * I_syn + input_current + rec_current
		next_V = (self.beta * V + next_I_syn) * (1.0 - Z.detach())
		next_Z = self.spike_func.apply(next_V, self.threshold, self.gamma)

		self._update_regularization_loss((next_V, next_I_syn, next_Z))
		return next_Z, (next_V, next_I_syn, next_Z)


class ALIFLayer(LIFLayer):
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			use_recurrent_connection: bool = True,
			use_rec_eye_mask: bool = True,
			spike_func: Type[SpikeFunction] = HeavisideSigmoidApprox,
			learning_type: LearningType = LearningType.BACKPROP,
			dt: float = 1e-3,
			device: Optional[torch.device] = None,
			**kwargs
	):
		super(ALIFLayer, self).__init__(
			input_size=input_size,
			output_size=output_size,
			name=name,
			use_recurrent_connection=use_recurrent_connection,
			use_rec_eye_mask=use_rec_eye_mask,
			spike_func=spike_func,
			learning_type=learning_type,
			dt=dt,
			device=device,
			**kwargs
		)
		self.beta = torch.tensor(self.kwargs["beta"], dtype=torch.float32, device=self.device)
		if self.kwargs["learn_beta"]:
			self.beta = torch.nn.Parameter(self.beta, requires_grad=True)
		self.rho = torch.tensor(np.exp(-dt / self.kwargs["tau_a"]), dtype=torch.float32, device=self.device)

	def _set_default_kwargs(self):
		self.kwargs.setdefault("tau_m", 20.0 * self.dt)
		self.kwargs.setdefault("tau_a", 200.0 * self.dt)
		self.kwargs.setdefault("beta", 1.6)
		self.kwargs.setdefault("threshold", 0.03)
		if isinstance(self.spike_func, HeavisideSigmoidApprox):
			self.kwargs.setdefault("gamma", 100.0)
		else:
			self.kwargs.setdefault("gamma", 0.3)
		self.kwargs.setdefault("learn_beta", False)

	def create_empty_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
		"""
		Create an empty state in the following form:
			[[membrane potential of shape (batch_size, self.output_size)]
			[current threshold of shape (batch_size, self.output_size)]
			[spikes of shape (batch_size, self.output_size)]]
		:param batch_size: The size of the current batch.
		:return: The current state.
		"""
		state = [torch.zeros(
			(batch_size, int(self._output_size)),
			device=self.device,
			dtype=torch.float32,
			requires_grad=True,
		) for _ in range(3)]
		return tuple(state)

	def forward(self, inputs: torch.Tensor, state: Tuple[torch.Tensor, ...] = None):
		assert inputs.ndim == 2
		batch_size, nb_features = inputs.shape
		V, a, Z = self._init_forward_state(state, batch_size)
		input_current = torch.matmul(inputs, self.forward_weights)
		if self.use_recurrent_connection:
			rec_current = torch.matmul(Z, torch.mul(self.recurrent_weights, self.rec_mask))
		else:
			rec_current = 0.0
		# v_j^{t+1} = \alpha * v_j^t + \sum_i W_{ji}*z_i^t + \sum_i W_{ji}^{in}x_i^{t+1} - z_j^t * v_{th}
		next_V = (self.alpha * V + input_current + rec_current) * (1.0 - Z.detach())
		next_a = self.rho * a + Z  # a^{t+1} = \rho * a_j^t + z_j^t
		A = self.threshold + self.beta * next_a  # A_j^t = v_{th} + \beta * a_j^t
		next_Z = self.spike_func.apply(next_V, A, self.gamma)  # z_j^t = H(v_j^t - A_j^t)
		return next_Z, (next_V, next_a, next_Z)


class IzhikevichLayer(BaseNeuronsLayer):
	"""
	Izhikevich p.274

	"""
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			use_recurrent_connection=True,
			use_rec_eye_mask=True,
			spike_func: Type[SpikeFunction] = HeavisideSigmoidApprox,
			learning_type: LearningType = LearningType.BACKPROP,
			dt=1e-3,
			device=None,
			**kwargs
	):
		self.spike_func = spike_func
		super(IzhikevichLayer, self).__init__(
			input_size=input_size,
			output_size=output_size,
			name=name,
			use_recurrent_connection=use_recurrent_connection,
			use_rec_eye_mask=use_rec_eye_mask,
			learning_type=learning_type,
			dt=dt,
			device=device,
			**kwargs
			)

		self.C = torch.tensor(self.kwargs["C"], dtype=torch.float32, device=self.device)
		self.v_rest = torch.tensor(self.kwargs["v_rest"], dtype=torch.float32, device=self.device)
		self.v_th = torch.tensor(self.kwargs["v_th"], dtype=torch.float32, device=self.device)
		self.k = torch.tensor(self.kwargs["k"], dtype=torch.float32, device=self.device)
		self.a = torch.tensor(self.kwargs["a"], dtype=torch.float32, device=self.device)
		self.b = torch.tensor(self.kwargs["b"], dtype=torch.float32, device=self.device)
		self.c = torch.tensor(self.kwargs["c"], dtype=torch.float32, device=self.device)
		self.d = torch.tensor(self.kwargs["d"], dtype=torch.float32, device=self.device)
		self.v_peak = torch.tensor(self.kwargs["v_peak"], dtype=torch.float32, device=self.device)
		self.gamma = torch.tensor(self.kwargs["gamma"], dtype=torch.float32, device=self.device)
		self.initialize_weights_()

	def _set_default_kwargs(self):
		self.kwargs.setdefault("C", 100.0)
		self.kwargs.setdefault("v_rest", -60.0)
		self.kwargs.setdefault("v_th", -40.0)
		self.kwargs.setdefault("k", 0.7)
		self.kwargs.setdefault("a", 0.03)
		self.kwargs.setdefault("b", -2.0)
		self.kwargs.setdefault("c", -50.0)
		self.kwargs.setdefault("d", 100.0)
		self.kwargs.setdefault("v_peak", 35.0)
		if isinstance(self.spike_func, HeavisideSigmoidApprox):
			self.kwargs.setdefault("gamma", 100.0)
		else:
			self.kwargs.setdefault("gamma", 1.0)

	def initialize_weights_(self):
		gain = 1.0
		for param in self.parameters():
			if param.ndim > 2:
				torch.nn.init.xavier_normal_(param, gain=gain)
			else:
				torch.nn.init.normal_(param, std=gain)

	def create_empty_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
		"""
		Create an empty state in the following form:
			([membrane potential of shape (batch_size, self.output_size)],
			[membrane potential of shape (batch_size, self.output_size)],
			[spikes of shape (batch_size, self.output_size)])
		:param batch_size: The size of the current batch.
		:return: The current state.
		"""
		V = self.v_rest * torch.ones(
			(batch_size, int(self._output_size)),
			device=self.device,
			dtype=torch.float32,
			requires_grad=True,
		)
		u = torch.zeros(
			(batch_size, int(self._output_size)),
			device=self.device,
			dtype=torch.float32,
			requires_grad=True,
		)
		Z = torch.zeros(
			(batch_size, int(self._output_size)),
			device=self.device,
			dtype=torch.float32,
			requires_grad=True,
		)
		return V, u, Z

	def forward(self, inputs: torch.Tensor, state: Tuple[torch.Tensor, ...] = None):
		assert inputs.ndim == 2
		batch_size, nb_features = inputs.shape
		V, u, Z = self._init_forward_state(state, batch_size)
		input_current = torch.matmul(inputs, self.forward_weights)
		if self.use_recurrent_connection:
			rec_current = torch.matmul(Z, torch.mul(self.recurrent_weights, self.rec_mask))
		else:
			rec_current = 0.0
		is_reset = Z.detach()
		I = input_current + rec_current
		dVdt = self.k * (V - self.v_rest) * (V - self.v_th) - u + I
		next_V = (V + self.dt * dVdt / self.C) * (1.0 - is_reset) + self.c * is_reset
		dudt = self.a * (self.b * (V - self.v_rest) - u)
		next_u = (u + self.dt * dudt) + self.d * is_reset
		next_Z = self.spike_func.apply(next_V, self.v_peak, self.gamma)
		return next_Z, (next_V, next_u, next_Z)


class WilsonCowanLayer(BaseNeuronsLayer):
	pass


class LILayer(BaseNeuronsLayer):
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			learning_type: LearningType = LearningType.BACKPROP,
			dt: float = 1e-3,
			device: Optional[torch.device] = None,
			**kwargs
	):
		super(LILayer, self).__init__(
			input_size=input_size,
			output_size=output_size,
			name=name,
			use_recurrent_connection=False,
			learning_type=learning_type,
			dt=dt,
			device=device,
			**kwargs
		)
		self.bias_weights = None
		self.kappa = torch.tensor(np.exp(-self.dt / self.kwargs["tau_out"]), dtype=torch.float32, device=self.device)

	def _set_default_kwargs(self):
		self.kwargs.setdefault("tau_out", 10.0 * self.dt)
		self.kwargs.setdefault("use_bias", True)

	def build(self):
		if self.kwargs["use_bias"]:
			self.bias_weights = nn.Parameter(
				torch.empty((int(self.output_size),), device=self.device),
				requires_grad=self.requires_grad,
			)
		else:
			self.bias_weights = torch.tensor(0.0, dtype=torch.float32, device=self.device)
		super(LILayer, self).build()
		self.initialize_weights_()

	def initialize_weights_(self):
		super(LILayer, self).initialize_weights_()
		torch.nn.init.constant_(self.bias_weights, 0.0)

	def create_empty_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
		"""
		Create an empty state in the following form:
			[membrane potential of shape (batch_size, self.output_size)]
		:param batch_size: The size of the current batch.
		:return: The current state.
		"""
		state = [torch.zeros(
			(batch_size, int(self._output_size)),
			device=self.device,
			dtype=torch.float32,
			requires_grad=True,
		)]
		return tuple(state)

	def forward(self, inputs: torch.Tensor, state: Tuple[torch.Tensor, ...] = None):
		assert inputs.ndim == 2
		batch_size, nb_features = inputs.shape
		# state = self._init_forward_state(state, batch_size)
		V, = self._init_forward_state(state, batch_size)
		next_V = self.kappa * V + torch.matmul(inputs, self.forward_weights) + self.bias_weights
		return next_V, (next_V, )


class SpyLILayer(BaseNeuronsLayer):
	"""
	The SpyLI layer is a LI layer implemented by the SpyTorch library
	(https://github.com/surrogate-gradient-learning/spytorch) from the
	paper: https://ieeexplore.ieee.org/document/8891809. In this LI varient they are using a second differential
	equation to model the synaptic current.
	"""
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			learning_type: LearningType = LearningType.BACKPROP,
			dt: float = 1e-3,
			device: Optional[torch.device] = None,
			**kwargs
	):
		super(SpyLILayer, self).__init__(
			input_size=input_size,
			output_size=output_size,
			name=name,
			use_recurrent_connection=False,
			learning_type=learning_type,
			dt=dt,
			device=device,
			**kwargs
		)
		self.bias_weights = None
		self.alpha = torch.tensor(np.exp(-dt / self.kwargs["tau_syn"]), dtype=torch.float32, device=self.device)
		self.beta = torch.tensor(np.exp(-dt / self.kwargs["tau_mem"]), dtype=torch.float32, device=self.device)

	def _set_default_kwargs(self):
		self.kwargs.setdefault("tau_syn", 5.0 * self.dt)
		self.kwargs.setdefault("tau_mem", 10.0 * self.dt)
		self.kwargs.setdefault("use_bias", False)

	def build(self):
		if self.kwargs["use_bias"]:
			self.bias_weights = nn.Parameter(
				torch.empty((int(self.output_size),), device=self.device),
				requires_grad=self.requires_grad,
			)
		else:
			self.bias_weights = torch.tensor(0.0, dtype=torch.float32, device=self.device)
		super(SpyLILayer, self).build()
		self.initialize_weights_()

	def initialize_weights_(self):
		super(SpyLILayer, self).initialize_weights_()
		weight_scale = 0.2
		torch.nn.init.normal_(self.forward_weights, mean=0.0, std=weight_scale / np.sqrt(int(self.input_size)))
		if self.kwargs["use_bias"]:
			torch.nn.init.constant_(self.bias_weights, 0.0)

	def create_empty_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
		"""
		Create an empty state in the following form:
			[membrane potential of shape (batch_size, self.output_size),
			synaptic current of shape (batch_size, self.output_size)]
		:param batch_size: The size of the current batch.
		:return: The current state.
		"""
		state = [torch.zeros(
			(batch_size, int(self._output_size)),
			device=self.device,
			dtype=torch.float32,
			requires_grad=True,
		) for _ in range(2)]
		return tuple(state)

	def forward(self, inputs: torch.Tensor, state: Tuple[torch.Tensor, ...] = None):
		assert inputs.ndim == 2
		batch_size, nb_features = inputs.shape
		V, I_syn = self._init_forward_state(state, batch_size)
		next_I_syn = self.alpha * I_syn + torch.matmul(inputs, self.forward_weights)
		next_V = self.beta * V + next_I_syn + self.bias_weights
		return next_V, (next_V, next_I_syn)


LayerType2Layer = {
	LayerType.LIF: LIFLayer,
	LayerType.ALIF: ALIFLayer,
	LayerType.Izhikevich: IzhikevichLayer,
	LayerType.LI: LILayer,
}

