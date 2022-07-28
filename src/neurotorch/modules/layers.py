import enum
import warnings
from copy import deepcopy
from typing import Any, List, Optional, Sized, Tuple, Type, Union, Iterable

import numpy as np
import torch
from torch import nn

from . import HeavisideSigmoidApprox, SpikeFunction
from ..dimension import Dimension, DimensionProperty, DimensionsLike, SizeTypes
from ..transforms import to_tensor


class LearningType(enum.Enum):
	NONE = 0
	BPTT = 1
	E_PROP = 2


class LayerType(enum.Enum):
	LIF = 0
	ALIF = 1
	Izhikevich = 2
	LI = 3
	SpyLIF = 4
	SpyLI = 5

	@classmethod
	def from_str(cls, name: str) -> Optional['LayerType']:
		"""
		Get the LayerType from a string.
		:param name: The name of the LayerType.
		:return: The LayerType.
		"""
		if isinstance(name, LayerType):
			return name
		if name.startswith(cls.__name__):
			name = name.removeprefix(f"{cls.__name__}.")
		if name not in cls.__members__:
			return None
		return cls[name]


class BaseLayer(torch.nn.Module):
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			learning_type: LearningType = LearningType.BPTT,
			device: Optional[torch.device] = None,
			**kwargs
	):
		super(BaseLayer, self).__init__()
		self._is_built = False
		self._name_is_set = False
		self.name = name
		self._name_is_default = name is None

		self.learning_type = learning_type
		self._device = device
		if self._device is None:
			self._set_default_device_()

		self.kwargs = kwargs
		self._set_default_kwargs()

		self.input_size = input_size
		self.output_size = output_size

		self._regularization_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
		self.reset_regularization_loss()

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
		return self.learning_type == LearningType.BPTT

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

	@property
	def device(self):
		return self._device

	@device.setter
	def device(self, device: torch.device):
		"""
		Set the device of the layer and move all the parameters to the new device.
		:param device: The device to set.
		:return: None
		"""
		self._device = device
		self.to(device)

	def __repr__(self):
		_repr = f"{self.__class__.__name__}"
		if self.name_is_set:
			_repr += f"<{self.name}>"
		_repr += f"({int(self.input_size)}->{int(self.output_size)})"
		_repr += f"[{self.learning_type}]"
		return _repr

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
		self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	def build(self):
		if self._is_built:
			raise ValueError("The layer can't be built multiple times.")
		if not self.is_ready_to_build:
			raise ValueError("Input size and output size must be specified before the build call.")
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
		inputs = inputs.to(self._device)
		if not self.is_built:
			if not self.is_ready_to_build:
				self.infer_sizes_from_inputs(inputs)
			self.build()
		call_output = super(BaseLayer, self).__call__(inputs, *args, **kwargs)

		if isinstance(call_output, torch.Tensor):
			hidden_state = None
		elif isinstance(call_output, (List, Tuple)) and len(call_output) == 2:
			hidden_state = call_output[1]
		else:
			raise ValueError(
				"The forward method must return a torch.Tensor (the output of the layer) "
				"or a tuple of torch.Tensor (the output of the layer and the hidden state)."
			)
		self.update_regularization_loss(hidden_state)
		return call_output

	def forward(self, inputs: torch.Tensor, state: torch.Tensor = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
		raise NotImplementedError()

	def initialize_weights_(self):
		for param in self.parameters():
			if param.ndim > 2:
				torch.nn.init.xavier_normal_(param)
			else:
				torch.nn.init.normal_(param)

	def update_regularization_loss(self, state: Optional[Any] = None, *args, **kwargs) -> torch.Tensor:
		"""
		Update the regularization loss for this layer. Each update call increments the regularization loss so at the end
		the regularization loss will be the sum of all calls to this function. This method is called at the end of each
		forward call automatically by the BaseLayer class.
		:param state: The current state of the layer.
		:param args: Other positional arguments.
		:param kwargs: Other keyword arguments.
		:return: The updated regularization loss.
		"""
		return self._regularization_loss

	def reset_regularization_loss(self):
		"""
		Reset the regularization loss to zero.
		:return: None
		"""
		self._regularization_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

	def get_and_reset_regularization_loss(self):
		"""
		Get and reset the regularization loss for this layer. The regularization loss will be reset by the
		reset_regularization_loss method after it is returned.
		:return: The regularization loss.
		"""
		loss = self.get_regularization_loss()
		self.reset_regularization_loss()
		return loss

	def get_regularization_loss(self):
		"""
		Get the regularization loss for this layer.
		:return: The regularization loss.
		"""
		return self._regularization_loss


class BaseNeuronsLayer(BaseLayer):
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			use_recurrent_connection: bool = True,
			use_rec_eye_mask: bool = False,
			learning_type: LearningType = LearningType.BPTT,
			dt: float = 1e-3,
			device: Optional[torch.device] = None,
			**kwargs
	):
		self.dt = dt
		self.use_recurrent_connection = use_recurrent_connection
		self.forward_weights = None
		self.use_rec_eye_mask = use_rec_eye_mask
		self.recurrent_weights = None
		self.rec_mask = None
		super().__init__(
			input_size=input_size,
			output_size=output_size,
			name=name,
			learning_type=learning_type,
			device=device,
			**kwargs
		)

	def create_empty_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
		raise NotImplementedError()

	def forward(self, inputs: torch.Tensor, state: torch.Tensor = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
		raise NotImplementedError()

	def initialize_weights_(self):
		super().initialize_weights_()
		if "forward_weights" in self.kwargs:
			self.forward_weights.data = to_tensor(self.kwargs["forward_weights"]).to(self.device)
		else:
			torch.nn.init.xavier_normal_(self.forward_weights)

		if "recurrent_weights" in self.kwargs and self.use_recurrent_connection:
			self.recurrent_weights.data = to_tensor(self.kwargs["recurrent_weights"]).to(self.device)
		elif self.use_recurrent_connection:
			torch.nn.init.xavier_normal_(self.recurrent_weights)

	def build(self):
		super().build()
		self.forward_weights = nn.Parameter(
			torch.empty((int(self.input_size), int(self.output_size)), device=self._device, dtype=torch.float32),
			requires_grad=self.requires_grad
		)
		if self.use_recurrent_connection:
			self.recurrent_weights = nn.Parameter(
				torch.empty((int(self.output_size), int(self.output_size)), device=self._device, dtype=torch.float32),
				requires_grad=self.requires_grad
			)
			if self.use_rec_eye_mask:
				self.rec_mask = (1 - torch.eye(int(self.output_size), device=self._device, dtype=torch.float32))
			else:
				self.rec_mask = torch.ones(
					(int(self.output_size), int(self.output_size)), device=self._device, dtype=torch.float32
				)
		self.initialize_weights_()


class LIFLayer(BaseNeuronsLayer):
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			use_recurrent_connection: bool = True,
			use_rec_eye_mask: bool = False,
			spike_func: Type[SpikeFunction] = HeavisideSigmoidApprox,
			learning_type: LearningType = LearningType.BPTT,
			dt: float = 1e-3,
			device: Optional[torch.device] = None,
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

	def _set_default_kwargs(self):
		self.kwargs.setdefault("tau_m", 10.0 * self.dt)
		self.kwargs.setdefault("threshold", 1.0)
		if issubclass(self.spike_func, HeavisideSigmoidApprox):
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

		if "forward_weights" in self.kwargs:
			self.forward_weights.data = to_tensor(self.kwargs["forward_weights"]).to(self.device)
		else:
			torch.nn.init.xavier_normal_(self.forward_weights)

		if "recurrent_weights" in self.kwargs and self.use_recurrent_connection:
			self.recurrent_weights.data = to_tensor(self.kwargs["recurrent_weights"]).to(self.device)
		elif self.use_recurrent_connection:
			torch.nn.init.xavier_normal_(self.recurrent_weights)

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
			device=self._device,
			dtype=torch.float32,
			requires_grad=True,
		) for _ in range(2)])
		return state

	def update_regularization_loss(self, state: Optional[Any] = None, *args, **kwargs) -> torch.Tensor:
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
			learning_type: LearningType = LearningType.BPTT,
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
		self.spike_func = HeavisideSigmoidApprox
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
		self._regularization_l1 = torch.tensor(0.0, dtype=torch.float32, device=self.device)
		self._n_spike_per_neuron = torch.zeros(int(self.output_size), dtype=torch.float32, device=self.device)
		self._total_count = 0

	def _set_default_kwargs(self):
		self.kwargs.setdefault("tau_syn", 5.0 * self.dt)
		self.kwargs.setdefault("tau_mem", 10.0 * self.dt)
		self.kwargs.setdefault("threshold", 1.0)
		self.kwargs.setdefault("gamma", 100.0)

	def initialize_weights_(self):
		weight_scale = 0.2
		if "forward_weights" in self.kwargs:
			self.forward_weights.data = to_tensor(self.kwargs["forward_weights"]).to(self.device)
		else:
			torch.nn.init.normal_(self.forward_weights, mean=0.0, std=weight_scale/np.sqrt(int(self.input_size)))

		if "recurrent_weights" in self.kwargs and self.use_recurrent_connection:
			self.recurrent_weights.data = to_tensor(self.kwargs["recurrent_weights"]).to(self.device)
		elif self.use_recurrent_connection:
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
			device=self._device,
			dtype=torch.float32,
			requires_grad=True,
		) for _ in range(3)])
		return state

	def reset_regularization_loss(self):
		super(SpyLIFLayer, self).reset_regularization_loss()
		self._regularization_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
		self._regularization_l1 = torch.tensor(0.0, dtype=torch.float32, device=self.device)
		self._n_spike_per_neuron = torch.zeros(int(self.output_size), dtype=torch.float32, device=self.device)
		self._total_count = 0

	def update_regularization_loss(self, state: Any, *args, **kwargs) -> torch.Tensor:
		"""
		Update the regularization loss for this layer. Each update call increments the regularization loss so at the end
		the regularization loss will be the sum of all calls to this function.
		:param state: The current state of the layer.
		:return: The updated regularization loss.
		"""
		next_V, next_I_syn, next_Z = state
		# self._regularization_l1 += 2e-6*torch.sum(next_Z)
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
		return next_Z, (next_V, next_I_syn, next_Z)


class ALIFLayer(LIFLayer):
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			use_recurrent_connection: bool = True,
			use_rec_eye_mask: bool = False,
			spike_func: Type[SpikeFunction] = HeavisideSigmoidApprox,
			learning_type: LearningType = LearningType.BPTT,
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
		self.beta = torch.tensor(self.kwargs["beta"], dtype=torch.float32, device=self._device)
		if self.kwargs["learn_beta"]:
			self.beta = torch.nn.Parameter(self.beta, requires_grad=True)
		self.rho = torch.tensor(np.exp(-dt / self.kwargs["tau_a"]), dtype=torch.float32, device=self._device)

	def _set_default_kwargs(self):
		self.kwargs.setdefault("tau_m", 20.0 * self.dt)
		self.kwargs.setdefault("tau_a", 200.0 * self.dt)
		self.kwargs.setdefault("beta", 1.6)
		# self.kwargs.setdefault("threshold", 0.03)
		self.kwargs.setdefault("threshold", 1.0)
		if issubclass(self.spike_func, HeavisideSigmoidApprox):
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
			device=self._device,
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

	def update_regularization_loss(self, state: Optional[Any] = None, *args, **kwargs) -> torch.Tensor:
		"""
		Update the regularization loss for this layer. Each update call increments the regularization loss so at the end
		the regularization loss will be the sum of all calls to this function.
		:param state: The current state of the layer.
		:return: The updated regularization loss.
		"""
		next_V, next_a, next_Z = state
		self._regularization_loss += 2e-6*torch.sum(next_Z)
		# self._regularization_loss += 2e-6*torch.mean(torch.sum(next_Z, dim=-1)**2)
		return self._regularization_loss


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
			learning_type: LearningType = LearningType.BPTT,
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

		self.C = torch.tensor(self.kwargs["C"], dtype=torch.float32, device=self._device)
		self.v_rest = torch.tensor(self.kwargs["v_rest"], dtype=torch.float32, device=self._device)
		self.v_th = torch.tensor(self.kwargs["v_th"], dtype=torch.float32, device=self._device)
		self.k = torch.tensor(self.kwargs["k"], dtype=torch.float32, device=self._device)
		self.a = torch.tensor(self.kwargs["a"], dtype=torch.float32, device=self._device)
		self.b = torch.tensor(self.kwargs["b"], dtype=torch.float32, device=self._device)
		self.c = torch.tensor(self.kwargs["c"], dtype=torch.float32, device=self._device)
		self.d = torch.tensor(self.kwargs["d"], dtype=torch.float32, device=self._device)
		self.v_peak = torch.tensor(self.kwargs["v_peak"], dtype=torch.float32, device=self._device)
		self.gamma = torch.tensor(self.kwargs["gamma"], dtype=torch.float32, device=self._device)
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
			device=self._device,
			dtype=torch.float32,
			requires_grad=True,
		)
		u = torch.zeros(
			(batch_size, int(self._output_size)),
			device=self._device,
			dtype=torch.float32,
			requires_grad=True,
		)
		Z = torch.zeros(
			(batch_size, int(self._output_size)),
			device=self._device,
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
	"""
	This layer is use for Wilson-Cowan neuronal dynamics.
	This dynamic is also referred to as firing rate model.
	Wilson-Cowan dynamic is great for neuronal calcium activity.
	This layer use recurrent neural network (RNN)
	The number of parameters that are trained is N^2 (+2N if mu and r is train)
	where N is the number of neurons.
	For references, please read:
	* Wilson HR, Cowan JD (1972) Excitatory and Inhibitory Interactions in
	Localized Populations of Model Neurons :
	https://doi.org/10.1016/S0006-3495(72)86068-5

	* Painchaud V, Doyon N, Desrosiers P (2022) Beyond Wilson-Cowan dynamics: oscillations
	and chaos without inhibitions :
	https://doi.org/10.48550/arXiv.2204.00583

	* Vogels TP, Rajan K, Abbott LF (2005) Neural Network dynamic:
	https://doi.org/10.1146/annurev.neuro.28.061604.135637

	The Wilson-Cowan dynamic is one of many dynamical models that can be used
	to model neuronal activity. To explore more continuous and Non-linear dynamics,
	please read:
	* Grossberg S (1987) Nonlinear Neural Network: Principles, Mechanisms, and Architecture :
	https://doi.org/10.1016/0893-6080(88)90021-4
	"""
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			learning_type: LearningType = LearningType.BPTT,
			dt: float = 1e-3,
			device=None,
			**kwargs
	):
		"""
		:param input_size: size of the input
		:param output_size: size of the output
			If we are predicting time series -> input_size = output_size
		:param learning_type: Type of learning for the gradient descent
		:param dt: Time step (Euler's discretisation)
		:param device: device for computation
		:param kwargs: Dict -> see below
		:keyword Arguments:
			* <std_weight>: float -> Instability of the initial random matrix
			* <mu>: float or torch.Tensor -> Activation threshold
				If torch.Tensor -> shape (1, number of neurons)
			* <tau>: float -> Decay constant of RNN unit
			* <learn_mu>: bool -> Whether to train the activation threshold
			* <mean_mu>: float -> Mean of the activation threshold (if learn_mu is True)
			* <std_mu>: float -> Standard deviation of the activation threshold (if learn_mu is True)
			* <r>: float or torch.Tensor -> Transition rate of the RNN unit
				If torch.Tensor -> shape (1, number of neurons)
			* <learn_r>: bool -> Whether to train the transition rate
			* <mean_r>: float -> Mean of the transition rate (if learn_r is True)
			* <std_r>: float -> Standard deviation of the transition rate (if learn_r is True)
			* <forward_weights>: torch.Tensor or np.ndarray -> set custom forward weights

		Remarks: Parameter mu and r can only be a parameter as a vector.
		"""
		super(WilsonCowanLayer, self).__init__(
			input_size=input_size,
			output_size=output_size,
			use_recurrent_connection=False,
			learning_type=learning_type,
			dt=dt,
			device=device,
			**kwargs
		)
		self.std_weight = self.kwargs["std_weight"]
		if not torch.is_tensor(self.kwargs["mu"]):
			self.mu = torch.tensor(self.kwargs["mu"], dtype=torch.float32, device=self._device)
		else:
			self.mu = self.kwargs["mu"]
			if self.mu.device != self._device:
				self.mu = self.mu.to(self._device)
			if self.mu.dtype != torch.float32:
				self.mu = self.mu.to(dtype=torch.float32)
		self.mean_mu = self.kwargs["mean_mu"]
		self.std_mu = self.kwargs["std_mu"]
		self.tau = self.kwargs["tau"]
		self.learn_mu = self.kwargs["learn_mu"]
		if not torch.is_tensor(self.kwargs["r"]):
			self.r = torch.tensor(self.kwargs["r"], dtype=torch.float32, device=self._device)
		else:
			self.r = self.kwargs["r"]
			if self.r.device != self._device:
				self.r = self.r.to(self._device)
			if self.r.dtype != torch.float32:
				self.r = self.r.to(dtype=torch.float32)
		self.mean_r = self.kwargs["mean_r"]
		self.std_r = self.kwargs["std_r"]
		self.learn_r = self.kwargs["learn_r"]

	def _set_default_kwargs(self):
		self.kwargs.setdefault("std_weight", 1.0)
		self.kwargs.setdefault("mu", 0.0)
		self.kwargs.setdefault("tau", 1.0)
		self.kwargs.setdefault("learn_mu", False)
		self.kwargs.setdefault("mean_mu", 2.0)
		self.kwargs.setdefault("std_mu", 0.0)
		self.kwargs.setdefault("r", 0.0)
		self.kwargs.setdefault("learn_r", False)
		self.kwargs.setdefault("mean_r", 2.0)
		self.kwargs.setdefault("std_r", 0.0)

	def initialize_weights_(self):
		"""
		Initialize the parameters (wights) that will be trained.
		"""
		if "forward_weights" in self.kwargs:
			self.forward_weights.data = to_tensor(self.kwargs["forward_weights"]).to(self.device)
		else:
			torch.nn.init.normal_(self.forward_weights, mean=0.0, std=self.std_weight)
		# If mu is not a parameter, it takes the value 0.0 unless stated otherwise by user
		# If mu is a parameter, it is initialized as a vector with the correct mean and std
		# unless stated otherwise by user.
		if self.learn_mu:
			if self.mu.dim() == 0:  # if mu is a scalar and a parameter -> convert it to a vector
				self.mu = torch.empty((1, self.forward_weights.shape[0]), dtype=torch.float32, device=self._device)
			self.mu = torch.nn.Parameter(self.mu, requires_grad=True)
			torch.nn.init.normal_(self.mu, mean=self.mean_mu, std=self.std_mu)
		if self.learn_r:
			if self.r.dim() == 0:
				self.r = torch.empty((1, self.forward_weights.shape[0]), dtype=torch.float32, device=self._device)
			self.r = torch.nn.Parameter(self.r, requires_grad=True)
			torch.nn.init.normal_(self.r, mean=self.mean_r, std=self.std_r)

	def create_empty_state(self, batch_size: int = 1) -> None:
		"""
		Create an empty state. With RNN, this function is not use
		:param batch_size: The size of the current batch
		:return: None
		"""
		return None

	def forward(self, inputs: torch.Tensor, state: torch.Tensor = None) -> Tuple[torch.Tensor, Tuple[None]]:
		"""
		Forward pass
		With Euler discretisation, Wilson-Cowan equation becomes:
		output = input * (1 - dt/tau) + dt/tau * (1 - r @ input) * sigmoid(forward_weights @ input - mu)
		:param inputs: time series at a time t of shape (batch_size, number of neurons)
			Remark: if use to compute a time series, use batch_size = 1
		:param state: State of the layer (only for SNN -> not use for RNN)
		:return: (time series at a time t+1, State of the layer -> None)
		"""
		ratio_dt_tau = self.dt / self.tau
		transition_rate = (1 - inputs * self.r)
		sigmoid = torch.sigmoid(torch.matmul(inputs, self.forward_weights) - self.mu)
		output = inputs * (1 - ratio_dt_tau) + transition_rate * sigmoid * ratio_dt_tau
		return output, (None, )


class LILayer(BaseNeuronsLayer):
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			learning_type: LearningType = LearningType.BPTT,
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
		self.kappa = torch.tensor(np.exp(-self.dt / self.kwargs["tau_out"]), dtype=torch.float32, device=self._device)

	def _set_default_kwargs(self):
		self.kwargs.setdefault("tau_out", 10.0 * self.dt)
		self.kwargs.setdefault("use_bias", True)

	def build(self):
		if self.kwargs["use_bias"]:
			self.bias_weights = nn.Parameter(
				torch.empty((int(self.output_size),), device=self._device),
				requires_grad=self.requires_grad,
			)
		else:
			self.bias_weights = torch.zeros((int(self.output_size),), dtype=torch.float32, device=self._device)
		super(LILayer, self).build()
		self.initialize_weights_()

	def initialize_weights_(self):
		super(LILayer, self).initialize_weights_()
		if "bias_weights" in self.kwargs:
			self.forward_weights.data = to_tensor(self.kwargs["bias_weights"]).to(self.device)
		else:
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
			device=self._device,
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
			learning_type: LearningType = LearningType.BPTT,
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
		self.alpha = torch.tensor(np.exp(-dt / self.kwargs["tau_syn"]), dtype=torch.float32, device=self._device)
		self.beta = torch.tensor(np.exp(-dt / self.kwargs["tau_mem"]), dtype=torch.float32, device=self._device)

	def _set_default_kwargs(self):
		self.kwargs.setdefault("tau_syn", 5.0 * self.dt)
		self.kwargs.setdefault("tau_mem", 10.0 * self.dt)
		self.kwargs.setdefault("use_bias", False)

	def build(self):
		if self.kwargs["use_bias"]:
			self.bias_weights = nn.Parameter(
				torch.empty((int(self.output_size),), device=self._device),
				requires_grad=self.requires_grad,
			)
		else:
			self.bias_weights = torch.tensor(0.0, dtype=torch.float32, device=self._device)
		super(SpyLILayer, self).build()
		self.initialize_weights_()

	def initialize_weights_(self):
		super(SpyLILayer, self).initialize_weights_()
		weight_scale = 0.2
		if "forward_weights" in self.kwargs:
			self.forward_weights.data = to_tensor(self.kwargs["forward_weights"]).to(self.device)
		else:
			torch.nn.init.normal_(self.forward_weights, mean=0.0, std=weight_scale/np.sqrt(int(self.input_size)))
		if self.kwargs["use_bias"]:
			if "bias_weights" in self.kwargs:
				self.forward_weights.data = to_tensor(self.kwargs["bias_weights"]).to(self.device)
			else:
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
			device=self._device,
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
	LayerType.SpyLIF: SpyLIFLayer,
	LayerType.SpyLI: SpyLILayer,
}

