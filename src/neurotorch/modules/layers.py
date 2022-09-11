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
from ..utils import inherit_method_docstring


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
		:type name: str
		
		:return: The LayerType.
		:rtype: Optional[LayerType]
		"""
		if isinstance(name, LayerType):
			return name
		if name.startswith(cls.__name__):
			name = name.removeprefix(f"{cls.__name__}.")
		if name not in cls.__members__:
			return None
		return cls[name]


class BaseLayer(torch.nn.Module):
	"""
	Base class for all layers.
	
	:Attributes:
		- **input_size** (Optional[Dimension]): The input size of the layer.
		- **output_size** (Optional[Dimension]): The output size of the layer.
		- **name** (str): The name of the layer.
		- **kwargs** (dict): Additional keyword arguments.
	
	"""
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			learning_type: LearningType = LearningType.BPTT,
			device: Optional[torch.device] = None,
			**kwargs
	):
		"""
		Constructor of the BaseLayer class.
		
		:param input_size: The input size of the layer.
		:type input_size: Optional[SizeTypes]
		:param output_size: The output size of the layer.
		:type output_size: Optional[SizeTypes]
		:param name: The name of the layer.
		:type name: Optional[str]
		:param learning_type: The learning type of the layer.
		:type learning_type: LearningType
		:param device: The device of the layer. Defaults to the current available device.
		:type device: Optional[torch.device]
		:param kwargs: Additional keyword arguments.
		
		:keyword bool regularize: Whether to regularize the layer. If True, the method `update_regularization_loss` will be
		called after each forward pass. Defaults to False.
		"""
		super(BaseLayer, self).__init__()
		self._is_built = False
		self._name_is_set = False
		self.name = name
		self._name_is_default = name is None

		self._learning_type = learning_type
		self._device = device
		if self._device is None:
			self._set_default_device_()

		self.kwargs = kwargs
		self._set_default_kwargs()

		self.input_size = input_size
		self.output_size = output_size

		self._regularization_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

	@property
	def input_size(self) -> Optional[Dimension]:
		if not hasattr(self, "_input_size"):
			return None
		return self._input_size

	@input_size.setter
	def input_size(self, size: Optional[SizeTypes]):
		self._input_size = self._format_size(size)

	@property
	def output_size(self) -> Optional[Dimension]:
		if not hasattr(self, "_output_size"):
			return None
		return self._output_size

	@output_size.setter
	def output_size(self, size: Optional[SizeTypes]):
		self._output_size = self._format_size(size)
	
	@property
	def learning_type(self):
		return self._learning_type
	
	@learning_type.setter
	def learning_type(self, learning_type: LearningType):
		self._learning_type = learning_type
		self.requires_grad_(self.requires_grad)
	
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
		:type device: torch.device
		
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
		_repr += f"@{self.device}"
		return _repr

	def _format_size(self, size: Optional[SizeTypes]) -> Optional[Dimension]:
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

	def build(self) -> 'BaseLayer':
		"""
		Build the layer. This method must be call after the layer is initialized to make sure that the layer is ready
		to be used e.g. the input and output size is set, the weights are initialized, etc.
		
		:return: The layer itself.
		:rtype: BaseLayer
		"""
		if self._is_built:
			raise ValueError("The layer can't be built multiple times.")
		if not self.is_ready_to_build:
			raise ValueError("Input size and output size must be specified before the build call.")
		self._is_built = True
		self.reset_regularization_loss()
		return self

	def create_empty_state(self, batch_size: int = 1, **kwargs) -> Tuple[torch.Tensor, ...]:
		"""
		Create an empty state for the layer. This method must be implemented by the child class.
		
		:param batch_size: The batch size of the state.
		:type batch_size: int
		
		:return: The empty state.
		:rtype: Tuple[torch.Tensor, ...]
		"""
		raise NotImplementedError()

	def _init_forward_state(
			self,
			state: Tuple[torch.Tensor, ...] = None,
			batch_size: int = 1,
			**kwargs
	) -> Tuple[torch.Tensor, ...]:
		if state is None:
			state = self.create_empty_state(batch_size, **kwargs)
		elif isinstance(state, (list, tuple)) and any([e is None for e in state]):
			empty_state = self.create_empty_state(batch_size, **kwargs)
			state = list(state)
			for i, e in enumerate(state):
				if e is None:
					state[i] = empty_state[i]
			state = tuple(state)
		return state

	def infer_sizes_from_inputs(self, inputs: torch.Tensor):
		"""
		Try to infer the input and output size of the layer from the inputs.
		
		:param inputs: The inputs to infer the size from.
		:type inputs: torch.Tensor
		
		:return: None
		"""
		self.input_size = inputs.shape[-1]
		if self.output_size is None:
			raise ValueError("output_size must be specified before the forward call.")
	
	def __call__(self, inputs: torch.Tensor, *args, **kwargs):
		"""
		Call the forward method of the layer. If the layer is not built, it will be built automatically.
		In addition, if :attr: `kwargs['regularize']` is set to True, the :meth: `update_regularization_loss` method
		will be called.
		
		:param inputs: The inputs to the layer.
		:type inputs: torch.Tensor
		:param args: The positional arguments to the forward method.
		:param kwargs: The keyword arguments to the forward method.
		
		:return: The output of the layer.
		"""
		inputs = inputs.to(self.device)
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
		if self.kwargs.get("regularize", False):
			self.update_regularization_loss(hidden_state)
		return call_output

	def forward(self, inputs: torch.Tensor, state: torch.Tensor = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
		raise NotImplementedError()

	def initialize_weights_(self):
		"""
		Initialize the weights of the layer. This method must be implemented by the child class.
		
		:return: None
		"""
		pass

	def update_regularization_loss(self, state: Optional[Any] = None, *args, **kwargs) -> torch.Tensor:
		"""
		Update the regularization loss for this layer. Each update call increments the regularization loss so at the end
		the regularization loss will be the sum of all calls to this function. This method is called at the end of each
		forward call automatically by the BaseLayer class.
		
		:param state: The current state of the layer.
		:type state: Optional[Any]
		:param args: Other positional arguments.
		:param kwargs: Other keyword arguments.
		
		:return: The updated regularization loss.
		:rtype: torch.Tensor
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
		
		WARNING: If this method is not called after an integration, the update of the regularization loss can cause a
		memory leak. TODO: fix this.
		
		:return: The regularization loss.
		"""
		loss = self.get_regularization_loss()
		self.reset_regularization_loss()
		return loss

	def get_regularization_loss(self) -> torch.Tensor:
		"""
		Get the regularization loss for this layer.
		
		:return: The regularization loss.
		"""
		return self._regularization_loss


class BaseNeuronsLayer(BaseLayer):
	"""
	A base class for layers that have neurons. This class provides two importants Parameters: the forward_weights and
	the recurrent_weights. Child classes must implement the forward method and the `create_empty_state` method.
	
	:Attributes:
		- **forward_weights** (torch.nn.Parameter): The weights used to compute the output of the layer.
		- **recurrent_weights** (torch.nn.Parameter): The weights used to compute the hidden state of the layer.
		- **dt** (float): The time step of the layer.
		- **use_rec_eye_mask** (torch.Tensor): Whether to use the recurrent eye mask.
		- **rec_mask** (torch.Tensor): The recurrent eye mask.
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
		Initialize the layer.
		
		:param input_size: The input size of the layer.
		:type input_size: Optional[SizeTypes]
		:param output_size: The output size of the layer.
		:type output_size: Optional[SizeTypes]
		:param name: The name of the layer.
		:type name: Optional[str]
		:param use_recurrent_connection: Whether to use a recurrent connection. Default is True.
		:type use_recurrent_connection: bool
		:param use_rec_eye_mask: Whether to use a recurrent eye mask. Default is False. This mask will be used to
			mask to zero the diagonal of the recurrent connection matrix.
		:type use_rec_eye_mask: bool
		:param learning_type: The learning type of the layer. Default is BPTT.
		:type learning_type: LearningType
		:param dt: The time step of the layer. Default is 1e-3.
		:type dt: float
		:param device: The device of the layer. Default is the current available device.
		:type device: Optional[torch.device]
		:param kwargs: Other keyword arguments.
		
		:keyword bool regularize: Whether to regularize the layer. If True, the method `update_regularization_loss` will be
		called after each forward pass. Defaults to False.
		"""
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

	def build(self) -> 'BaseNeuronsLayer':
		"""
		Build the layer. This method must be call after the layer is initialized to make sure that the layer is ready
		to be used e.g. the input and output size is set, the weights are initialized, etc.
		
		In this method the :attr:`forward_weights`, :attr:`recurrent_weights` and :attr: `rec_mask` are created and
		finally the method :meth:`initialize_weights_` is called.
		
		:return: The layer itself.
		:rtype: BaseLayer
		"""
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
				self.rec_mask = nn.Parameter(
					(1 - torch.eye(int(self.output_size), device=self.device, dtype=torch.float32)),
					requires_grad=False
				)
			else:
				self.rec_mask = nn.Parameter(
					torch.ones(
						(int(self.output_size), int(self.output_size)), device=self.device, dtype=torch.float32
					),
					requires_grad=False
				)
		self.initialize_weights_()
		return self
	
	def __repr__(self):
		_repr = f"{self.__class__.__name__}"
		if self.name_is_set:
			_repr += f"<{self.name}>"
		_repr += f"({int(self.input_size)}"
		if self.use_recurrent_connection:
			_repr += "<"
		_repr += f"->{int(self.output_size)})"
		_repr += f"[{self.learning_type}]"
		_repr += f"@{self.device}"
		return _repr


class LIFLayer(BaseNeuronsLayer):
	"""
	LIF dynamics, inspired by :cite:t:`neftci_surrogate_2019` , :cite:t:`bellec_solution_2020` , models the synaptic
	potential and impulses of a neuron over time. The shape of this potential is not considered realistic
	:cite:t:`izhikevich_dynamical_2007` , but the time at which the potential exceeds the threshold is.
	This potential is found by the recurrent equation :eq:`lif_V` .
	
	.. math::
		:label: lif_V
		
		V_j^{t+\\Delta t} = \\left(\\alpha V_j^t + \\sum_{i}^{N} W_{ij}^{\\text{rec}} z_i^t +
		\\sum_i^{N} W_{ij}^{\\text{in}} x_i^{t+\\Delta t}\\right) \\left(1 - z_j^t\\right)
	
	The variables of the equation :eq:`lif_V` are described by the following definitions:
		
		- :math:`N` is the number of neurons in the layer.
		- :math:`V_j^t` is the synaptic potential of the neuron :math:`j` at time :math:`t`.
		- :math:`\\Delta t` is the integration time step.
		- :math:`z_j^t` is the spike of the neuron :math:`j` at time :math:`t`.
		- :math:`\\alpha` is the decay constant of the potential over time (equation :eq:`lif_alpha` ).
		- :math:`W_{ij}^{\\text{rec}}` is the recurrent weight of the neuron :math:`i` to the neuron :math:`j`.
		- :math:`W_{ij}^{\\text{in}}` is the input weight of the neuron :math:`i` to the neuron :math:`j`.
		- :math:`x_i^{t}` is the input of the neuron :math:`i` at time :math:`t`.
	
	.. math::
		:label: lif_alpha
		
		\\alpha = e^{-\\frac{\\Delta t}{\\tau_m}}
	
	with :math:`\\tau_m` being the decay time constant of the membrane potential which is generally 20 ms.
	
	The output of neuron :math:`j` at time :math:`t` denoted :math:`z_j^t` is defined by the equation :eq:`lif_z` .
	
	.. math::
		:label: lif_z
		
		z_j^t = H(V_j^t - V_{\\text{th}})
	
	where :math:``V_{\\text{th}} denotes the activation threshold of the neuron and the function :math:`H(\\cdot)`
	is the Heaviside function defined as :math:`H(x) = 1` if :math:`x \\geq 0` and :math:`H(x) = 0` otherwise.

	.. bibliography::
	
	:Attributes:
		- :attr:`forward_weights` (torch.nn.Parameter): The weights used to compute the output of the layer :math:`W_{ij}^{\\text{in}}` in equation :eq:`lif_V`.
		- :attr:`recurrent_weights` (torch.nn.Parameter): The weights used to compute the hidden state of the layer :math:`W_{ij}^{\\text{rec}}` in equation :eq:`lif_V`.
		- :attr:`dt` (float): The time step of the layer :math:`\\Delta t` in equation :eq:`lif_V`.
		- :attr:`use_rec_eye_mask` (bool): Whether to use the recurrent eye mask.
		- :attr:`rec_mask` (torch.Tensor): The recurrent eye mask.
		- :attr:`alpha` (torch.nn.Parameter): The decay constant of the potential over time. See equation :eq:`lif_alpha` .
		- :attr:`threshold` (torch.nn.Parameter): The activation threshold of the neuron.
		- :attr:`gamma` (torch.nn.Parameter): The gain of the neuron. The gain will increase the gradient of the neuron's output.
	
	"""
	
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
		"""
		:keyword float tau_m: The decay time constant of the membrane potential which is generally 20 ms. See equation
			:eq:`lif_alpha` .
		:keyword float threshold: The activation threshold of the neuron.
		:keyword float gamma: The gain of the neuron. The gain will increase the gradient of the neuron's output.
		:keyword float spikes_regularization_factor: The regularization factor of the spikes.
		"""
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

		self.alpha = nn.Parameter(
			torch.tensor(np.exp(-dt / self.kwargs["tau_m"]), dtype=torch.float32, device=self.device),
			requires_grad=False
		)
		self.threshold = nn.Parameter(
			torch.tensor(self.kwargs["threshold"], dtype=torch.float32, device=self.device),
			requires_grad=False
		)
		self.gamma = nn.Parameter(
			torch.tensor(self.kwargs["gamma"], dtype=torch.float32, device=self.device),
			requires_grad=False
		)

	def _set_default_kwargs(self):
		self.kwargs.setdefault("tau_m", 10.0 * self.dt)
		self.kwargs.setdefault("threshold", 1.0)
		if issubclass(self.spike_func, HeavisideSigmoidApprox):
			self.kwargs.setdefault("gamma", 100.0)
		else:
			self.kwargs.setdefault("gamma", 1.0)
		self.kwargs.setdefault("spikes_regularization_factor", 0.0)

	def initialize_weights_(self):
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
		self._regularization_loss += self.kwargs["spikes_regularization_factor"]*torch.sum(next_Z)
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

		:param input_size: The size of the input.
		:param output_size: The size of the output.
		:param name: The name of the layer.
		:param use_recurrent_connection: Whether to use the recurrent connection.
		:param use_rec_eye_mask: Whether to use the recurrent eye mask.
		:param spike_func: The spike function to use.
		:param learning_type: The learning type to use.
		:param dt: Time step (Euler's discretisation).
		:param device: The device to use.
		:param kwargs: The keyword arguments for the layer.

		:Keyword Arguments:
			* <tau_syn>: float -> The synaptic time constant. Default: 5.0 * dt.
			* <tau_mem>: float -> The membrane time constant. Default: 10.0 * dt.
			* <threshold>: float -> The threshold of the layer. Default: 1.0.
			* <gamma>: float -> The multiplier of the derivative of the spike function. Default: 100.0.
			* <spikes_regularization_factor>: float -> The regularization factor for the spikes. Higher this factor is,
			the more the network will tend to spike less. Default: 0.0.
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

		self.alpha = nn.Parameter(
			torch.tensor(np.exp(-dt / self.kwargs["tau_syn"]), dtype=torch.float32, device=self.device),
			requires_grad=False
		)
		self.beta = nn.Parameter(
			torch.tensor(np.exp(-dt / self.kwargs["tau_mem"]), dtype=torch.float32, device=self.device),
			requires_grad=False
		)
		self.threshold = nn.Parameter(
			torch.tensor(self.kwargs["threshold"], dtype=torch.float32, device=self.device),
			requires_grad=False
		)
		self.gamma = nn.Parameter(
			torch.tensor(self.kwargs["gamma"], dtype=torch.float32, device=self.device),
			requires_grad=False
		)
		self._regularization_l1 = torch.tensor(0.0, dtype=torch.float32, device=self.device)
		self._n_spike_per_neuron = torch.zeros(int(self.output_size), dtype=torch.float32, device=self.device)
		self._total_count = 0

	def _set_default_kwargs(self):
		self.kwargs.setdefault("tau_syn", 5.0 * self.dt)
		self.kwargs.setdefault("tau_mem", 10.0 * self.dt)
		self.kwargs.setdefault("threshold", 1.0)
		self.kwargs.setdefault("gamma", 100.0)
		self.kwargs.setdefault("spikes_regularization_factor", 0.0)

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

	def update_regularization_loss(self, state: Optional[Any] = None, *args, **kwargs) -> torch.Tensor:
		"""
		Update the regularization loss for this layer. Each update call increments the regularization loss so at the end
		the regularization loss will be the sum of all calls to this function.
		
		:param state: The current state of the layer.
		:return: The updated regularization loss.
		"""
		next_V, next_I_syn, next_Z = state
		self._regularization_l1 += self.kwargs["spikes_regularization_factor"]*torch.sum(next_Z)
		# self._n_spike_per_neuron += torch.sum(torch.sum(next_Z, dim=0), dim=0)
		# self._total_count += next_Z.shape[0]*next_Z.shape[1]
		# current_l2 = self.kwargs["spikes_regularization_factor"]*torch.sum(self._n_spike_per_neuron ** 2) / (self._total_count + 1e-6)
		# self._regularization_loss = self._regularization_l1 + current_l2
		self._regularization_loss = self._regularization_l1
		return self._regularization_loss

	def forward(self, inputs: torch.Tensor, state: Tuple[torch.Tensor, ...] = None):
		assert inputs.ndim == 2, f"Inputs must be of shape (batch_size, input_size), got {inputs.shape}."
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
		self.beta = nn.Parameter(
			torch.tensor(self.kwargs["beta"], dtype=torch.float32, device=self.device),
			requires_grad=self.kwargs["learn_beta"]
		)
		self.rho = nn.Parameter(
			torch.tensor(np.exp(-dt / self.kwargs["tau_a"]), dtype=torch.float32, device=self.device),
			requires_grad=False
		)

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
		self.kwargs.setdefault("spikes_regularization_factor", 0.0)

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
		self._regularization_loss += self.kwargs["spikes_regularization_factor"]*torch.sum(next_Z)
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
			use_recurrent_connection: bool = False,
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
			* <forward_weights>: torch.Tensor or np.ndarray -> set custom forward weights
			* <std_weight>: float -> Instability of the initial random matrix
			* <mu>: float or torch.Tensor -> Activation threshold
				If torch.Tensor -> shape (1, number of neurons)
			* <mean_mu>: float -> Mean of the activation threshold (if learn_mu is True)
			* <std_mu>: float -> Standard deviation of the activation threshold (if learn_mu is True)
			* <learn_mu>: bool -> Whether to train the activation threshold
			* <tau>: float -> Decay constant of RNN unit
			* <learn_tau> -> bool -> Wheter to train the decay constant
			* <r>: float or torch.Tensor -> Transition rate of the RNN unit
				If torch.Tensor -> shape (1, number of neurons)
			* <mean_r>: float -> Mean of the transition rate (if learn_r is True)
			* <std_r>: float -> Standard deviation of the transition rate (if learn_r is True)
			* <learn_r>: bool -> Whether to train the transition rate

		Remarks: Parameter mu and r can only be a parameter as a vector.
		"""
		super(WilsonCowanLayer, self).__init__(
			input_size=input_size,
			output_size=output_size,
			use_recurrent_connection=use_recurrent_connection,
			learning_type=learning_type,
			dt=dt,
			device=device,
			**kwargs
		)
		self.std_weight = self.kwargs["std_weight"]
		self.mu = to_tensor(self.kwargs["mu"]).to(self.device)
		self.mean_mu = self.kwargs["mean_mu"]
		self.std_mu = self.kwargs["std_mu"]
		self.learn_mu = self.kwargs["learn_mu"]
		self.tau = to_tensor(self.kwargs["tau"]).to(self.device)
		self.learn_tau = self.kwargs["learn_tau"]
		self.r_sqrt = torch.sqrt(to_tensor(self.kwargs["r"], dtype=torch.float32)).to(self.device)
		self.mean_r = self.kwargs["mean_r"]
		self.std_r = self.kwargs["std_r"]
		self.learn_r = self.kwargs["learn_r"]

	def _set_default_kwargs(self):
		self.kwargs.setdefault("std_weight", 1.0)
		self.kwargs.setdefault("mu", 0.0)
		self.kwargs.setdefault("tau", 1.0)
		self.kwargs.setdefault("learn_tau", False)
		self.kwargs.setdefault("learn_mu", False)
		self.kwargs.setdefault("mean_mu", 2.0)
		self.kwargs.setdefault("std_mu", 0.0)
		self.kwargs.setdefault("r", 0.0)
		self.kwargs.setdefault("learn_r", False)
		self.kwargs.setdefault("mean_r", 2.0)
		self.kwargs.setdefault("std_r", 0.0)
		self.kwargs.setdefault("hh_init", "inputs")

	def _assert_kwargs(self):
		assert self.std_weight >= 0.0, "std_weight must be greater or equal to 0.0"
		assert self.std_mu >= 0.0, "std_mu must be greater or equal to 0.0"
		assert self.tau > 0.0, "tau must be greater than 0.0"
		assert self.tau > self.dt, "tau must be greater than dt"

	@property
	def r(self):
		"""
		This property is used to ensure that the transition rate will never be negative if trained.
		"""
		return self.r_sqrt**2

	def initialize_weights_(self):
		"""
		Initialize the parameters (weights) that will be trained.
		"""
		if "forward_weights" in self.kwargs:
			self.forward_weights.data = to_tensor(self.kwargs["forward_weights"]).to(self.device)
		else:
			torch.nn.init.normal_(self.forward_weights, mean=0.0, std=self.std_weight)

		if "recurrent_weights" in self.kwargs and self.use_recurrent_connection:
			self.recurrent_weights.data = to_tensor(self.kwargs["recurrent_weights"]).to(self.device)
		elif self.use_recurrent_connection:
			torch.nn.init.xavier_normal_(self.recurrent_weights)

		# If mu is not a parameter, it takes the value 0.0 unless stated otherwise by user
		# If mu is a parameter, it is initialized as a vector with the correct mean and std
		# unless stated otherwise by user.
		if self.learn_mu:
			if self.mu.dim() == 0:  # if mu is a scalar and a parameter -> convert it to a vector
				self.mu = torch.empty((1, self.forward_weights.shape[0]), dtype=torch.float32, device=self.device)
			self.mu = torch.nn.Parameter(self.mu, requires_grad=self.requires_grad)
			torch.nn.init.normal_(self.mu, mean=self.mean_mu, std=self.std_mu)
		if self.learn_r:
			_r = torch.empty((1, self.forward_weights.shape[0]), dtype=torch.float32, device=self.device)
			torch.nn.init.normal_(_r, mean=self.mean_r, std=self.std_r)
			self.r_sqrt = torch.nn.Parameter(torch.sqrt(torch.abs(_r)), requires_grad=self.requires_grad)
		if self.learn_tau:
			self.tau = torch.nn.Parameter(self.tau, requires_grad=self.requires_grad)

	def create_empty_state(self, batch_size: int = 1, **kwargs) -> Tuple[torch.Tensor]:
		"""
		Create an empty state. With RNN, this function is not use
		:param batch_size: The size of the current batch
		:return: None
		"""
		if self.kwargs["hh_init"] == "zeros":
			state = [torch.zeros(
				(batch_size, int(self.output_size)),
				device=self.device,
				dtype=torch.float32,
				requires_grad=True,
			) for _ in range(1)]
		elif self.kwargs["hh_init"] == "random":
			mu, std = self.kwargs.get("hh_init_mu", 0.0), self.kwargs.get("hh_init_std", 1.0)
			gen = torch.Generator()
			gen.manual_seed(self.kwargs.get("hh_init_seed", 0))
			state = [(torch.rand(
				(batch_size, int(self.output_size)),
				device=self.device,
				dtype=torch.float32,
				requires_grad=True,
				generator=gen,
			)*std + mu) for _ in range(1)]
		elif self.kwargs["hh_init"] == "inputs":
			assert "inputs" in kwargs, "inputs must be provided to initialize the state"
			assert kwargs["inputs"].shape == (batch_size, int(self.output_size))
			state = (kwargs["inputs"].clone(), )
		else:
			raise ValueError("Hidden state init method not known. Please use 'zeros', 'inputs' or 'random'")
		return tuple(state)

	def forward(
			self,
			inputs: torch.Tensor,
			state: Optional[torch.Tensor] = None
	) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
		"""
		Forward pass
		With Euler discretisation, Wilson-Cowan equation becomes:
		output = input * (1 - dt/tau) + dt/tau * (1 - input @ r) * sigmoid(input @ forward_weight - mu)
		:param inputs: time series at a time t of shape (batch_size, number of neurons)
			Remark: if you use to compute a time series, use batch_size = 1
		:param state: State of the layer (only for SNN -> not use for RNN)
		:return: (time series at a time t+1, State of the layer -> None)
		"""
		batch_size, nb_features = inputs.shape
		hh, = self._init_forward_state(state, batch_size, inputs=inputs)
		ratio_dt_tau = self.dt / self.tau

		if self.use_recurrent_connection:
			rec_inputs = torch.matmul(hh, torch.mul(self.recurrent_weights, self.rec_mask))
		else:
			rec_inputs = 0.0

		transition_rate = (1 - hh * self.r)
		sigmoid = torch.sigmoid(rec_inputs + torch.matmul(inputs, self.forward_weights) - self.mu)
		output = hh * (1 - ratio_dt_tau) + transition_rate * sigmoid * ratio_dt_tau
		return output, (output, )


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

	def build(self) -> 'LILayer':
		if self.kwargs["use_bias"]:
			self.bias_weights = nn.Parameter(
				torch.empty((int(self.output_size),), device=self._device),
				requires_grad=self.requires_grad,
			)
		else:
			self.bias_weights = torch.zeros((int(self.output_size),), dtype=torch.float32, device=self._device)
		super(LILayer, self).build()
		self.initialize_weights_()
		return self

	def initialize_weights_(self):
		super(LILayer, self).initialize_weights_()
		if "bias_weights" in self.kwargs:
			self.bias_weights.data = to_tensor(self.kwargs["bias_weights"]).to(self.device)
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

	def build(self) -> 'SpyLILayer':
		if self.kwargs["use_bias"]:
			self.bias_weights = nn.Parameter(
				torch.empty((int(self.output_size),), device=self._device),
				requires_grad=self.requires_grad,
			)
		else:
			self.bias_weights = torch.tensor(0.0, dtype=torch.float32, device=self._device)
		super(SpyLILayer, self).build()
		self.initialize_weights_()
		return self

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

