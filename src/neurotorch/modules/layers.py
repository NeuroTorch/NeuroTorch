import enum
import time
import warnings
from copy import deepcopy
from typing import Any, List, Optional, Sized, Tuple, Type, Union, Iterable
import inspect

import numpy as np
import torch
from torch import nn

from . import HeavisideSigmoidApprox, SpikeFunction, HeavisidePhiApprox
from ..dimension import Dimension, DimensionProperty, DimensionsLike, SizeTypes
from ..transforms import to_tensor, ToDevice
from pythonbasictools.docstring import inherit_docstring, inherit_fields_docstring

from ..utils import format_pseudo_rn_seed


class LayerType(enum.Enum):
	LIF = 0
	ALIF = 1
	Izhikevich = 2
	LI = 3
	SpyLIF = 4
	SpyLI = 5
	SpyALIF = 6

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
		- :attr:`input_size` (Optional[Dimension]): The input size of the layer.
		- :attr:`output_size` (Optional[Dimension]): The output size of the layer.
		- :attr:`name` (str): The name of the layer.
		- :attr:`kwargs` (dict): Additional keyword arguments.
	
	"""
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
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
		:param learning_type: The learning type of the layer. Deprecated use freeze_weights instead.
		:type learning_type: LearningType
		:param device: The device of the layer. Defaults to the current available device.
		:type device: Optional[torch.device]
		:param kwargs: Additional keyword arguments.
		
		:keyword bool regularize: Whether to regularize the layer. If True, the method `update_regularization_loss`
			will be called after each forward pass. Defaults to False.
		:keyword bool freeze_weights: Whether to freeze the weights of the layer. Defaults to False.
		"""
		super(BaseLayer, self).__init__()
		self._is_built = False
		self._name_is_set = False
		self.name = name
		self._name_is_default = name is None

		self._freeze_weights = kwargs.get("freeze_weights", False)
		self._device = device
		if self._device is None:
			self._set_default_device_()
		self._device_transform = ToDevice(self.device)

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
	def freeze_weights(self) -> bool:
		return self._freeze_weights
	
	@freeze_weights.setter
	def freeze_weights(self, freeze_weights: bool):
		self._freeze_weights = freeze_weights
		self.requires_grad_(self.requires_grad)
	
	@property
	def requires_grad(self):
		return not self.freeze_weights

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
		self.to(device, non_blocking=True)
		self._device_transform = ToDevice(device)

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
	
	def __call__(self, inputs: torch.Tensor, state: torch.Tensor = None, *args, **kwargs):
		"""
		Call the forward method of the layer. If the layer is not built, it will be built automatically.
		In addition, if :attr:`kwargs['regularize']` is set to True, the :meth: `update_regularization_loss` method
		will be called.
		
		:param inputs: The inputs to the layer.
		:type inputs: torch.Tensor
		
		:param args: The positional arguments to the forward method.
		:param kwargs: The keyword arguments to the forward method.
		
		:return: The output of the layer.
		"""
		inputs, state = self._device_transform(inputs), self._device_transform(state)
		if not self.is_built:
			if not self.is_ready_to_build:
				self.infer_sizes_from_inputs(inputs)
			self.build()
		call_output = super(BaseLayer, self).__call__(inputs, state, *args, **kwargs)

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

	def forward(
			self,
			inputs: torch.Tensor,
			state: torch.Tensor = None,
			**kwargs
	) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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


# @inherit_fields_docstring(fields=["Attributes"], bases=[BaseLayer])
class BaseNeuronsLayer(BaseLayer):
	"""
	A base class for layers that have neurons. This class provides two importants Parameters: the
	:attr:`forward_weights` and the :attr:`recurrent_weights`. Child classes must implement the :method:`forward`
	method and the :mth:`create_empty_state` method.
	
	:Attributes:
		- :attr:`forward_weights` (torch.nn.Parameter): The weights used to compute the output of the layer.
		- :attr:`recurrent_weights` (torch.nn.Parameter): The weights used to compute the hidden state of the layer.
		- :attr:`dt` (float): The time step of the layer.
		- :attr:`use_rec_eye_mask` (torch.Tensor): Whether to use the recurrent eye mask.
		- :attr:`rec_mask` (torch.Tensor): The recurrent eye mask.
	"""
	
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			use_recurrent_connection: bool = True,
			use_rec_eye_mask: bool = False,
			dt: float = 1e-3,
			device: Optional[torch.device] = None,
			**kwargs
	):
		"""
		Initialize the layer.; See the :class:`BaseLayer` class for more details.;
		
		:param input_size: The input size of the layer;
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
		:param kwargs: Other keyword arguments.
		
		:keyword bool regularize: Whether to regularize the layer. If True, the method `update_regularization_loss` will
			be called after each forward pass. Defaults to False.
		:keyword str hh_init: The initialization method for the hidden state. Defaults to "zeros".
		:keyword float hh_init_mu: The mean of the hidden state initialization when hh_init is random . Defaults to 0.0.
		:keyword float hh_init_std: The standard deviation of the hidden state initialization when hh_init is random. Defaults to 1.0.
		:keyword int hh_init_seed: The seed of the hidden state initialization when hh_init is random. Defaults to 0.
		:keyword bool force_dale_law: Whether to force the Dale's law in the layer's weights. Defaults to False.
		:keyword Union[torch.Tensor, float] forward_sign: If force_dale_law is True, this parameter will be used to
			initialize the forward_sign vector. If it is a float, the forward_sign vector will be initialized with this
			value as the ration of inhibitory neurons. If it is a tensor, it will be used as the forward_sign vector.
		:keyword Union[torch.Tensor, float] recurrent_sign: If force_dale_law is True, this parameter will be used to
			initialize the recurrent_sign vector. If it is a float, the recurrent_sign vector will be initialized with
			this value as the ration of inhibitory neurons. If it is a tensor, it will be used as the recurrent_sign vector.
		:keyword Callable sign_activation: The activation function used to compute the sign of the weights i.e. the
			forward_sign and recurrent_sign vectors. Defaults to torch.nn.Tanh.
		"""
		self.dt = dt
		self.use_recurrent_connection = use_recurrent_connection
		self._forward_weights = None
		self._forward_sign = None
		self.use_rec_eye_mask = use_rec_eye_mask
		self._recurrent_weights = None
		self._recurrent_sign = None
		self.rec_mask = None
		self._force_dale_law = kwargs.get("force_dale_law", False)
		super().__init__(
			input_size=input_size,
			output_size=output_size,
			name=name,
			device=device,
			**kwargs
		)
		self.sign_activation = self.kwargs.get("sign_activation", torch.nn.Tanh())
	
	@property
	def forward_weights(self) -> torch.nn.Parameter:
		"""
		Get the forward weights.
		
		:return: The forward weights.
		"""
		if self.force_dale_law:
			return torch.pow(self._forward_weights, 2) * self.forward_sign
		return self._forward_weights
	
	@forward_weights.setter
	def forward_weights(self, value: torch.nn.Parameter):
		"""
		Set the forward weights.
		
		:param value: The forward weights.
		"""
		if not isinstance(value, torch.nn.Parameter):
			value = torch.nn.Parameter(value, requires_grad=self.requires_grad)
		self._forward_weights = value
	
	@property
	def recurrent_weights(self) -> torch.nn.Parameter:
		"""
		Get the recurrent weights.
		
		:return: The recurrent weights.
		"""
		if self.force_dale_law:
			return torch.pow(self._recurrent_weights, 2) * self.recurrent_sign
		return self._recurrent_weights
	
	@recurrent_weights.setter
	def recurrent_weights(self, value: torch.nn.Parameter):
		"""
		Set the recurrent weights.
		
		:param value: The recurrent weights.
		"""
		if not isinstance(value, torch.nn.Parameter):
			value = torch.nn.Parameter(value, requires_grad=self.requires_grad)
		self._recurrent_weights = value
	
	@property
	def force_dale_law(self) -> bool:
		"""
		Get whether to force the Dale's law.
		
		:return: Whether to force the Dale's law.
		"""
		return self._force_dale_law
	
	@property
	def forward_sign(self) -> Optional[torch.nn.Parameter]:
		"""
		Get the forward sign.
		
		:return: The forward sign.
		"""
		if self._forward_sign is None:
			return None
		return self.sign_activation(self._forward_sign)
	
	@forward_sign.setter
	def forward_sign(self, value: torch.nn.Parameter):
		"""
		Set the forward sign.
		
		:param value: The forward sign.
		"""
		if not isinstance(value, torch.nn.Parameter):
			value = torch.nn.Parameter(value, requires_grad=self.force_dale_law and self.requires_grad)
		self._forward_sign = value

	@property
	def recurrent_sign(self) -> Optional[torch.nn.Parameter]:
		"""
		Get the recurrent sign.
		
		:return: The recurrent sign.
		"""
		if self._recurrent_sign is None:
			return None
		return self.sign_activation(self._recurrent_sign)
	
	@recurrent_sign.setter
	def recurrent_sign(self, value: torch.nn.Parameter):
		"""
		Set the recurrent sign.
		
		:param value: The recurrent sign.
		"""
		if not isinstance(value, torch.nn.Parameter):
			value = torch.nn.Parameter(value, requires_grad=self.force_dale_law)
		self._recurrent_sign = value
		
	def get_weights_parameters(self) -> List[torch.nn.Parameter]:
		"""
		Get the weights parameters.
		
		:return: The weights parameters.
		"""
		parameters = [self._forward_weights]
		if self.use_recurrent_connection:
			parameters.append(self._recurrent_weights)
		return parameters
	
	def get_sign_parameters(self) -> List[torch.nn.Parameter]:
		"""
		Get the sign parameters.
		
		:return: The sign parameters.
		"""
		parameters = []
		if self.force_dale_law:
			parameters.append(self._forward_sign)
			if self.use_recurrent_connection:
				parameters.append(self._recurrent_sign)
		return parameters

	def create_empty_state(
			self,
			batch_size: int = 1,
			**kwargs
	) -> Tuple[torch.Tensor, ...]:
		self.kwargs.setdefault("hh_init", "zeros")
		self.kwargs.setdefault("hh_init_mu", 0.0)
		self.kwargs.setdefault("hh_init_std", 1.0)
		
		n_hh = kwargs.get("n_hh", 1)
		if self.kwargs["hh_init"] == "zeros":
			state = tuple(
				[torch.zeros(
					(batch_size, int(self.output_size)),
					device=self._device,
					dtype=torch.float32,
					requires_grad=True,
				) for _ in range(n_hh)]
			)
		elif self.kwargs["hh_init"] == "random":
			mu, std = self.kwargs.get("hh_init_mu", 0.0), self.kwargs.get("hh_init_std", 1.0)
			gen = torch.Generator(device=self.device)
			gen.manual_seed(format_pseudo_rn_seed(self.kwargs.get("hh_init_seed", None)))
			state = [(torch.rand(
				(batch_size, int(self.output_size)),
				device=self.device,
				dtype=torch.float32,
				requires_grad=True,
				generator=gen,
			) * std + mu) for _ in range(n_hh)]
		elif self.kwargs["hh_init"] == "inputs":
			assert "inputs" in kwargs, "inputs must be provided to initialize the state"
			assert kwargs["inputs"].shape == (batch_size, int(self.output_size))
			state = [kwargs["inputs"].clone() for _ in range(n_hh)]
		else:
			raise ValueError("Hidden state init method not known. Please use 'zeros', 'inputs' or 'random'")
		return tuple(state)

	def forward(
			self,
			inputs: torch.Tensor,
			state: torch.Tensor = None,
			**kwargs
	) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
		raise NotImplementedError()

	def _init_forward_sign_(self):
		if self.kwargs.get("forward_sign", None) is None:
			self.kwargs.pop("forward_sign", None)
		if "forward_sign" in self.kwargs and self.force_dale_law:
			if isinstance(self.kwargs["forward_sign"], float):
				assert 0.0 <= self.kwargs["forward_sign"] <= 1.0, "forward_sign must be in [0, 1]"
				n_inh = int(int(self.input_size) * self.kwargs["forward_sign"])
				inh_indexes = torch.randperm(int(self.input_size))[:n_inh]
				self.kwargs["forward_sign"] = np.abs(np.random.normal(size=(int(self.input_size), 1)))
				self.kwargs["forward_sign"][inh_indexes] *= -1
			assert self.kwargs["forward_sign"].shape == (int(self.input_size), 1), \
				"forward_sign must be a float or a tensor of shape (input_size, 1)"
			self._forward_sign.data = to_tensor(self.kwargs["forward_sign"]).to(self.device)
			with torch.no_grad():
				self._forward_weights.data = torch.sqrt(torch.abs(self._forward_weights.data))
		elif self.force_dale_law:
			torch.nn.init.normal_(self._forward_sign)

	def _init_recurrent_sign_(self):
		if self.kwargs.get("recurrent_sign", None) is None:
			self.kwargs.pop("recurrent_sign", None)
		if "recurrent_sign" in self.kwargs and self.force_dale_law and self.use_recurrent_connection:
			if isinstance(self.kwargs["recurrent_sign"], float):
				assert 0.0 <= self.kwargs["recurrent_sign"] <= 1.0, "recurrent_sign must be in [0, 1]"
				n_inh = int(int(self.output_size) * self.kwargs["recurrent_sign"])
				inh_indexes = torch.randperm(int(self.output_size))[:n_inh]
				self.kwargs["recurrent_sign"] = np.abs(np.random.normal(size=(int(self.output_size), 1)))
				self.kwargs["recurrent_sign"][inh_indexes] *= -1
			assert self.kwargs["recurrent_sign"].shape == (int(self.output_size), 1), \
				"recurrent_sign must be a float or a tensor of shape (output_size, 1)"
			self._recurrent_sign.data = to_tensor(self.kwargs["recurrent_sign"]).to(self.device)
			with torch.no_grad():
				self._recurrent_weights.data = torch.sqrt(torch.abs(self._recurrent_weights.data))
		elif self.force_dale_law and self.use_recurrent_connection:
			torch.nn.init.xavier_normal_(self._recurrent_sign)

	def initialize_weights_(self):
		super().initialize_weights_()
		if "forward_weights" in self.kwargs:
			self._forward_weights.data = to_tensor(self.kwargs["forward_weights"]).to(self.device)
		else:
			torch.nn.init.xavier_normal_(self._forward_weights)

		if "recurrent_weights" in self.kwargs and self.use_recurrent_connection:
			self._recurrent_weights.data = to_tensor(self.kwargs["recurrent_weights"]).to(self.device)
		elif self.use_recurrent_connection:
			torch.nn.init.xavier_normal_(self._recurrent_weights)

		self._init_forward_sign_()
		self._init_recurrent_sign_()

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
		self._forward_weights = nn.Parameter(
			torch.empty((int(self.input_size), int(self.output_size)), device=self.device, dtype=torch.float32),
			requires_grad=self.requires_grad
		)
		if self.force_dale_law:
			self._forward_sign = torch.nn.Parameter(
				torch.empty((int(self.input_size), 1), dtype=torch.float32, device=self.device),
				requires_grad=self.force_dale_law
			)

		if self.use_recurrent_connection:
			self._recurrent_weights = nn.Parameter(
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
			if self.force_dale_law:
				self._recurrent_sign = torch.nn.Parameter(
					torch.empty((int(self.output_size), 1), dtype=torch.float32, device=self.device),
					requires_grad=self.force_dale_law
				)
		self.initialize_weights_()
		return self
	
	def __repr__(self):
		_repr = f"{self.__class__.__name__}"
		if self.name_is_set:
			_repr += f"<{self.name}>"
		if self.force_dale_law:
			_repr += f"[Dale]"
		_repr += f"({int(self.input_size)}"
		if self.use_recurrent_connection:
			_repr += "<"
		_repr += f"->{int(self.output_size)})"
		if self.freeze_weights:
			_repr += "[frozen]"
		_repr += f"@{self.device}"
		return _repr


class Linear(BaseNeuronsLayer):
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			device: Optional[torch.device] = None,
			**kwargs
	):
		super().__init__(
			input_size=input_size,
			output_size=output_size,
			name=name,
			use_recurrent_connection=False,
			device=device,
			**kwargs
		)
		self.bias_weights = None
		self.activation = self._init_activation(self.kwargs["activation"])
	
	def _set_default_kwargs(self):
		self.kwargs.setdefault("use_bias", True)
		self.kwargs.setdefault("activation", "identity")
	
	def _init_activation(self, activation: Union[torch.nn.Module, str]):
		"""
		Initialise the activation function.

		:param activation: Activation function.
		:type activation: Union[torch.nn.Module, str]
		"""
		str_to_activation = {
			"identity": torch.nn.Identity(),
			"relu"    : torch.nn.ReLU(),
			"tanh"    : torch.nn.Tanh(),
			"sigmoid" : torch.nn.Sigmoid(),
		}
		if isinstance(activation, str):
			assert activation in str_to_activation.keys(), f"Activation {activation} is not implemented."
			self.activation = str_to_activation[activation]
		else:
			self.activation = activation
		return self.activation
	
	def build(self) -> 'Linear':
		if self.kwargs["use_bias"]:
			self.bias_weights = nn.Parameter(
				torch.empty((int(self.output_size),), device=self._device),
				requires_grad=self.requires_grad,
			)
		else:
			self.bias_weights = torch.zeros((int(self.output_size),), dtype=torch.float32, device=self._device)
		super().build()
		self.initialize_weights_()
		return self
	
	def initialize_weights_(self):
		super().initialize_weights_()
		if "bias_weights" in self.kwargs:
			self.bias_weights.data = to_tensor(self.kwargs["bias_weights"]).to(self.device)
		else:
			torch.nn.init.constant_(self.bias_weights, 0.0)
	
	def create_empty_state(
			self,
			batch_size: int = 1,
			**kwargs
	) -> Tuple[torch.Tensor, ...]:
		kwargs.setdefault("n_hh", 0)
		return super().create_empty_state(batch_size=batch_size, **kwargs)
	
	def forward(
			self,
			inputs: torch.Tensor,
			state: Tuple[torch.Tensor, ...] = None,
			**kwargs
	):
		# assert inputs.ndim == 2
		# batch_size, nb_features = inputs.shape
		return self.activation(torch.matmul(inputs, self.forward_weights) + self.bias_weights)


# @inherit_fields_docstring(fields=["Attributes"], bases=[BaseNeuronsLayer])
class LIFLayer(BaseNeuronsLayer):
	"""
	LIF dynamics, inspired by :cite:t:`neftci_surrogate_2019` , :cite:t:`bellec_solution_2020` , models the synaptic
	potential and impulses of a neuron over time. The shape of this potential is not considered realistic
	:cite:t:`izhikevich_dynamical_2007` , but the time at which the potential exceeds the threshold is.
	This potential is found by the recurrent equation :eq:`lif_V` .
	
	.. math::
		\\begin{equation}
			V_j^{t+\\Delta t} = \\left(\\alpha V_j^t + \\sum_{i}^{N} W_{ij}^{\\text{rec}} z_i^t +
			\\sum_i^{N} W_{ij}^{\\text{in}} x_i^{t+\\Delta t}\\right) \\left(1 - z_j^t\\right)
		\\end{equation}
		:label: lif_V
		
	
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
		
		\\begin{equation}
			\\alpha = e^{-\\frac{\\Delta t}{\\tau_m}}
		\\end{equation}
		
		
	
	with :math:`\\tau_m` being the decay time constant of the membrane potential which is generally 20 ms.
	
	The output of neuron :math:`j` at time :math:`t` denoted :math:`z_j^t` is defined by the equation :eq:`lif_z` .
	
	.. math::
		:label: lif_z
		
		z_j^t = H(V_j^t - V_{\\text{th}})
	
	where :math:`V_{\\text{th}}` denotes the activation threshold of the neuron and the function :math:`H(\\cdot)`
	is the Heaviside function defined as :math:`H(x) = 1` if :math:`x \\geq 0` and :math:`H(x) = 0` otherwise.
	
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
	
	# @inherit_docstring(bases=BaseNeuronsLayer)
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			use_recurrent_connection: bool = True,
			use_rec_eye_mask: bool = False,
			spike_func: Type[SpikeFunction] = HeavisideSigmoidApprox,
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
	
	# @inherit_docstring(bases=BaseNeuronsLayer)
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

	def create_empty_state(
			self,
			batch_size: int = 1,
			**kwargs
	) -> Tuple[torch.Tensor, ...]:
		"""
		Create an empty state in the following form:
			([membrane potential of shape (batch_size, self.output_size)],
			[spikes of shape (batch_size, self.output_size)])
			
		:param batch_size: The size of the current batch.
		:return: The current state.
		"""
		kwargs.setdefault("n_hh", 2)
		return super(LIFLayer, self).create_empty_state(batch_size=batch_size, **kwargs)

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
	
	# @inherit_docstring(bases=BaseNeuronsLayer)
	def forward(
			self,
			inputs: torch.Tensor,
			state: Tuple[torch.Tensor, ...] = None,
			**kwargs
	):
		assert inputs.ndim == 2
		batch_size, nb_features = inputs.shape
		V, Z = self._init_forward_state(state, batch_size, inputs=inputs)
		input_current = torch.matmul(inputs, self.forward_weights)
		if self.use_recurrent_connection:
			rec_current = torch.matmul(Z, torch.mul(self.recurrent_weights, self.rec_mask))
		else:
			rec_current = 0.0
		next_V = (self.alpha * V + input_current + rec_current) * (1.0 - Z.detach())
		next_Z = self.spike_func.apply(next_V, self.threshold, self.gamma)
		return next_Z, (next_V, next_Z)


# @inherit_fields_docstring(fields=["Attributes"], bases=[BaseNeuronsLayer])
class SpyLIFLayer(BaseNeuronsLayer):
	"""
	The SpyLIF dynamics is a more complex variant of the LIF dynamics (class :class:`LIFLayer`) allowing it to have a
	greater power of expression. This variant is also inspired by Neftci :cite:t:`neftci_surrogate_2019` and also
	contains  two differential equations like the SpyLI dynamics :class:`SpyLI`. The equation :eq:`SpyLIF_I` presents
	the synaptic current update equation with euler integration while the equation :eq:`SpyLIF_V` presents the
	synaptic potential update.
	
	.. math::
		:label: SpyLIF_I
		
		\\begin{equation}
			I_{\\text{syn}, j}^{t+\\Delta t} = \\alpha I_{\text{syn}, j}^{t} + \\sum_{i}^{N} W_{ij}^{\\text{rec}} z_i^t
			+ \\sum_i^{N} W_{ij}^{\\text{in}} x_i^{t+\\Delta t}
		\\end{equation}
		
		
	.. math::
		:label: SpyLIF_V
		
		\\begin{equation}
			V_j^{t+\\Delta t} = \\left(\\beta V_j^t + I_{\\text{syn}, j}^{t+\\Delta t}\\right) \\left(1 - z_j^t\\right)
		\\end{equation}
		
		
	.. math::
		:label: spylif_alpha
		
		\\begin{equation}
			\\alpha = e^{-\\frac{\\Delta t}{\\tau_{\\text{syn}}}}
		\\end{equation}
	
	with :math:`\\tau_{\\text{syn}}` being the decay time constant of the synaptic current.
	
	.. math::
		:label: spylif_beta
		
		\\begin{equation}
			\\beta = e^{-\\frac{\\Delta t}{\\tau_{\\text{mem}}}}
		\\end{equation}
	
	with :math:`\\tau_{\\text{syn}}` being the decay time constant of the synaptic current.
	
	The output of neuron :math:`j` at time :math:`t` denoted :math:`z_j^t` is defined by the equation :eq:`spylif_z` .
	
	.. math::
		:label: spylif_z
		
		z_j^t = H(V_j^t - V_{\\text{th}})
	
	where :math:`V_{\\text{th}}` denotes the activation threshold of the neuron and the function :math:`H(\\cdot)`
	is the Heaviside function defined as :math:`H(x) = 1` if :math:`x \\geq 0` and :math:`H(x) = 0` otherwise.
	
	SpyTorch library: https://github.com/surrogate-gradient-learning/spytorch.
	
	The variables of the equations :eq:`SpyLIF_I` and :eq:`SpyLIF_V` are described by the following definitions:
		
		- :math:`N` is the number of neurons in the layer.
		- :math:`I_{\\text{syn}, j}^{t}` is the synaptic current of neuron :math:`j` at time :math:`t`.
		- :math:`V_j^t` is the synaptic potential of the neuron :math:`j` at time :math:`t`.
		- :math:`\\Delta t` is the integration time step.
		- :math:`z_j^t` is the spike of the neuron :math:`j` at time :math:`t`.
		- :math:`\\alpha` is the decay constant of the synaptic current over time (equation :eq:`spylif_alpha`).
		- :math:`\\beta` is the decay constant of the membrane potential over time (equation :eq:`spylif_beta`).
		- :math:`W_{ij}^{\\text{rec}}` is the recurrent weight of the neuron :math:`i` to the neuron :math:`j`.
		- :math:`W_{ij}^{\\text{in}}` is the input weight of the neuron :math:`i` to the neuron :math:`j`.
		- :math:`x_i^{t}` is the input of the neuron :math:`i` at time :math:`t`.
	
	:Attributes:
		- :attr:`alpha` (torch.nn.Parameter): Decay constant of the synaptic current over time (equation :eq:`spylif_alpha`).
		- :attr:`beta` (torch.nn.Parameter): Decay constant of the membrane potential over time (equation :eq:`spylif_beta`).
		- :attr:`threshold` (torch.nn.Parameter): Activation threshold of the neuron (:math:`V_{\\text{th}}`).
		- :attr:`gamma` (torch.nn.Parameter): Slope of the Heaviside function (:math:`\\gamma`).
	"""
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			use_recurrent_connection: bool = True,
			use_rec_eye_mask: bool = False,
			dt: float = 1e-3,
			device: Optional[torch.device] = None,
			**kwargs
	):
		"""
		Constructor for the SpyLIF layer.

		:param input_size: The size of the input.
		:type input_size: Optional[SizeTypes]
		:param output_size: The size of the output.
		:type output_size: Optional[SizeTypes]
		:param name: The name of the layer.
		:type name: Optional[str]
		:param use_recurrent_connection: Whether to use the recurrent connection.
		:type use_recurrent_connection: bool
		:param use_rec_eye_mask: Whether to use the recurrent eye mask.
		:type use_rec_eye_mask: bool
		:param spike_func: The spike function to use.
		:type spike_func: Callable[[torch.Tensor], torch.Tensor]
		:param learning_type: The learning type to use.
		:type learning_type: LearningType
		:param dt: Time step (Euler's discretisation).
		:type dt: float
		:param device: The device to use.
		:type device: Optional[torch.device]
		:param kwargs: The keyword arguments for the layer.

		:keyword float tau_syn: The synaptic time constant :math:`\\tau_{\\text{syn}}`. Default: 5.0 * dt.
		:keyword float tau_mem: The membrane time constant :math:`\\tau_{\\text{mem}}`. Default: 10.0 * dt.
		:keyword float threshold: The threshold potential :math:`V_{\\text{th}}`. Default: 1.0.
		:keyword float gamma: The multiplier of the derivative of the spike function :math:`\\gamma`. Default: 100.0.
		:keyword float spikes_regularization_factor: The regularization factor for the spikes. Higher this factor is,
			the more the network will tend to spike less. Default: 0.0.

		"""
		self.spike_func = HeavisideSigmoidApprox
		super(SpyLIFLayer, self).__init__(
			input_size=input_size,
			output_size=output_size,
			name=name,
			use_recurrent_connection=use_recurrent_connection,
			use_rec_eye_mask=use_rec_eye_mask,
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
		self.kwargs.setdefault("hh_init", "zeros")

	def initialize_weights_(self):
		super().initialize_weights_()
		weight_scale = 0.2
		if "forward_weights" in self.kwargs:
			self.forward_weights.data = to_tensor(self.kwargs["forward_weights"]).to(self.device)
		else:
			torch.nn.init.normal_(self.forward_weights, mean=0.0, std=weight_scale/np.sqrt(int(self.input_size)))

		if "recurrent_weights" in self.kwargs and self.use_recurrent_connection:
			self.recurrent_weights.data = to_tensor(self.kwargs["recurrent_weights"]).to(self.device)
		elif self.use_recurrent_connection:
			torch.nn.init.normal_(self.recurrent_weights, mean=0.0, std=weight_scale/np.sqrt(int(self.output_size)))

	def create_empty_state(
			self,
			batch_size: int = 1,
			**kwargs
	) -> Tuple[torch.Tensor, ...]:
		"""
		Create an empty state in the following form:
			([membrane potential of shape (batch_size, self.output_size)],
			[synaptic current of shape (batch_size, self.output_size)],
			[spikes of shape (batch_size, self.output_size)])
		
		:param batch_size: The size of the current batch.
		:return: The current state.
		"""
		kwargs.setdefault("n_hh", 3)
		thr = self.threshold.detach().cpu().item()
		if self.kwargs["hh_init"] == "random":
			V_mu, V_std = self.kwargs.get("hh_init_mu", thr / 2.0), self.kwargs.get("hh_init_std", 0.341 * thr)
			gen = torch.Generator(device=self.device)
			gen.manual_seed(format_pseudo_rn_seed(self.kwargs.get("hh_init_seed", None)))
			V = torch.clamp_min(torch.rand(
				(batch_size, int(self.output_size)),
				device=self.device,
				dtype=torch.float32,
				requires_grad=True,
				generator=gen,
			) * V_std + V_mu, min=0.0)
			I = torch.rand(
				(batch_size, int(self.output_size)),
				device=self.device,
				dtype=torch.float32,
				requires_grad=True,
				generator=gen,
			)
			Z = self.spike_func.apply(V, self.threshold, self.gamma)
			V = V * (1.0 - Z)
			return tuple([V, I, Z])
		elif self.kwargs["hh_init"] == "inputs":
			assert "inputs" in kwargs, "The inputs must be provided to initialize the state."
			assert int(self.input_size) == int(self.output_size), \
				"The input and output size must be the same with inputs initialization."
			# V_mu, V_std = self.kwargs.get("hh_init_mu", thr / 2.0), self.kwargs.get("hh_init_std", 0.341 * thr)
			V_mu, V_std = self.kwargs.get("hh_init_mu", 0.0), self.kwargs.get("hh_init_std", thr)
			gen = torch.Generator(device=self.device)
			gen.manual_seed(format_pseudo_rn_seed(self.kwargs.get("hh_init_seed", None)))
			I = torch.rand(
				(batch_size, int(self.output_size)),
				device=self.device,
				dtype=torch.float32,
				requires_grad=True,
				generator=gen,
			)
			Z = kwargs["inputs"].clone()
			V = (torch.rand(
				(batch_size, int(self.output_size)),
				device=self.device,
				dtype=torch.float32,
				requires_grad=True,
				generator=gen,
			) * V_std + V_mu)
			V = (self.beta * V + self.alpha * I) * (1.0 - Z)
			
			return tuple([V, I, Z])
		return super(SpyLIFLayer, self).create_empty_state(batch_size=batch_size, **kwargs)

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

	def forward(
			self,
			inputs: torch.Tensor,
			state: Tuple[torch.Tensor, ...] = None,
			**kwargs
	):
		assert inputs.ndim == 2, f"Inputs must be of shape (batch_size, input_size), got {inputs.shape}."
		batch_size, nb_features = inputs.shape
		V, I_syn, Z = self._init_forward_state(state, batch_size, inputs=inputs)
		input_current = torch.matmul(inputs, self.forward_weights)
		if self.use_recurrent_connection:
			rec_current = torch.matmul(Z, torch.mul(self.recurrent_weights, self.rec_mask))
		else:
			rec_current = 0.0
		next_I_syn = self.alpha * I_syn + input_current + rec_current
		next_V = (self.beta * V + next_I_syn) * (1.0 - Z.detach())
		next_Z = self.spike_func.apply(next_V, self.threshold, self.gamma)
		return next_Z, (next_V, next_I_syn, next_Z)


class SpyALIFLayer(SpyLIFLayer):
	"""
	The SpyALIF dynamic, inspired by Bellec and \\textit{al.} :cite:t:`bellec_solution_2020` and bye the
	:class:`SpyLIFLayer` from the work of Neftci :cite:t:`neftci_surrogate_2019`, is very
	similar to the SpyLIF dynamics (class :class:`SpyLIFLayer`). In fact, SpyALIF has exactly the same potential
	update equation as SpyLIF. The difference comes
	from the fact that the threshold potential varies with time and neuron input. Indeed, the threshold
	is increased at each output spike and is then decreased with a certain rate in order to come back to
	its starting threshold :math:`V_{\\text{th}}`. The threshold equation from :class:`SpyLIFLayer` is thus slightly
	modified by changing :math:`V_{\\text{th}} \\to A_j^t`. Thus, the output of neuron :math:`j` at time :math:`t`
	denoted :math:`z_j^t` is redefined by the equation :eq:`alif_z`.

	.. math::
		:label: SpyALIF_I

		\\begin{equation}
			I_{\\text{syn}, j}^{t+\\Delta t} = \\alpha I_{\text{syn}, j}^{t} + \\sum_{i}^{N} W_{ij}^{\\text{rec}} z_i^t
			+ \\sum_i^{N} W_{ij}^{\\text{in}} x_i^{t+\\Delta t}
		\\end{equation}


	.. math::
		:label: SpyALIF_V

		\\begin{equation}
			V_j^{t+\\Delta t} = \\left(\\beta V_j^t + I_{\\text{syn}, j}^{t+\\Delta t}\\right) \\left(1 - z_j^t\\right)
		\\end{equation}


	.. math::
		:label: spyalif_alpha

		\\begin{equation}
			\\alpha = e^{-\\frac{\\Delta t}{\\tau_{\\text{syn}}}}
		\\end{equation}

	with :math:`\\tau_{\\text{syn}}` being the decay time constant of the synaptic current.

	.. math::
		:label: spyalif_beta

		\\begin{equation}
			\\beta = e^{-\\frac{\\Delta t}{\\tau_{\\text{mem}}}}
		\\end{equation}

	with :math:`\\tau_{\\text{syn}}` being the decay time constant of the synaptic current.

	The output of neuron :math:`j` at time :math:`t` denoted :math:`z_j^t` is defined by the equation :eq:`spyalif_z` .

	.. math::
		:label: spyalif_z

		z_j^t = H(V_j^t - A_j^t)

	where :math:`A_j^t` denotes the activation threshold of the neuron and the function :math:`H(\\cdot)`
	is the Heaviside function defined as :math:`H(x) = 1` if :math:`x \\geq 0` and :math:`H(x) = 0` otherwise.
	The update of the activation threshold is then described by :eq:`alif_A`.
	
	.. math::
		:label: alif_A
	
		\\begin{equation}
			A_j^t = V_{\\text{th}} + \\kappa a_j^t
		\\end{equation}
	
	with the adaptation variable :math:`a_j^t` described by :eq:`alif_a` and :math:`\\kappa` an amplification
	factor greater than 1 and typically equivalent to :math:`\\kappa\\approx 1.6` :cite:t:`bellec_solution_2020`.
	
	.. math::
		:label: alif_a
	
		\\begin{equation}
			a_j^{t+1} = \\rho a_j + z_j^t
		\\end{equation}
	
	With the decay factor :math:`\\rho` as:
	
	.. math::
		:label: alif_rho
		
		\\begin{equation}
			\\rho = e^{-\\frac{\\Delta t}{\\tau_a}}
		\\end{equation}

	SpyTorch library: https://github.com/surrogate-gradient-learning/spytorch.

	The variables of the equations :eq:`SpyALIF_I` and :eq:`SpyALIF_V` are described by the following definitions:

		- :math:`N` is the number of neurons in the layer.
		- :math:`I_{\\text{syn}, j}^{t}` is the synaptic current of neuron :math:`j` at time :math:`t`.
		- :math:`V_j^t` is the synaptic potential of the neuron :math:`j` at time :math:`t`.
		- :math:`\\Delta t` is the integration time step.
		- :math:`z_j^t` is the spike of the neuron :math:`j` at time :math:`t`.
		- :math:`\\alpha` is the decay constant of the synaptic current over time (equation :eq:`spylif_alpha`).
		- :math:`\\beta` is the decay constant of the membrane potential over time (equation :eq:`spylif_beta`).
		- :math:`W_{ij}^{\\text{rec}}` is the recurrent weight of the neuron :math:`i` to the neuron :math:`j`.
		- :math:`W_{ij}^{\\text{in}}` is the input weight of the neuron :math:`i` to the neuron :math:`j`.
		- :math:`x_i^{t}` is the input of the neuron :math:`i` at time :math:`t`.

	:Attributes:
		- :attr:`alpha` (torch.nn.Parameter): Decay constant of the synaptic current over time (equation :eq:`spyalif_alpha`).
		- :attr:`beta` (torch.nn.Parameter): Decay constant of the membrane potential over time (equation :eq:`spyalif_beta`).
		- :attr:`threshold` (torch.nn.Parameter): Activation threshold of the neuron (:math:`V_{\\text{th}}`).
		- :attr:`gamma` (torch.nn.Parameter): Slope of the Heaviside function (:math:`\\gamma`).
		- :attr:`kappa`: The amplification factor of the threshold potential (:math:`\\kappa`).
		- :attr:`rho`: The decay factor of the adaptation variable (:math:`\\rho`).
	"""
	
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			use_recurrent_connection: bool = True,
			use_rec_eye_mask: bool = False,
			dt: float = 1e-3,
			device: Optional[torch.device] = None,
			**kwargs
	):
		"""
		Constructor for the SpyLIF layer.

		:param input_size: The size of the input.
		:type input_size: Optional[SizeTypes]
		:param output_size: The size of the output.
		:type output_size: Optional[SizeTypes]
		:param name: The name of the layer.
		:type name: Optional[str]
		:param use_recurrent_connection: Whether to use the recurrent connection.
		:type use_recurrent_connection: bool
		:param use_rec_eye_mask: Whether to use the recurrent eye mask.
		:type use_rec_eye_mask: bool
		:param spike_func: The spike function to use.
		:type spike_func: Callable[[torch.Tensor], torch.Tensor]
		:param learning_type: The learning type to use.
		:type learning_type: LearningType
		:param dt: Time step (Euler's discretisation).
		:type dt: float
		:param device: The device to use.
		:type device: Optional[torch.device]
		:param kwargs: The keyword arguments for the layer.

		:keyword float tau_syn: The synaptic time constant :math:`\\tau_{\\text{syn}}`. Default: 5.0 * dt.
		:keyword float tau_mem: The membrane time constant :math:`\\tau_{\\text{mem}}`. Default: 10.0 * dt.
		:keyword float threshold: The threshold potential :math:`V_{\\text{th}}`. Default: 1.0.
		:keyword float gamma: The multiplier of the derivative of the spike function :math:`\\gamma`. Default: 100.0.
		:keyword float spikes_regularization_factor: The regularization factor for the spikes. Higher this factor is,
			the more the network will tend to spike less. Default: 0.0.

		"""
		self.spike_func = HeavisideSigmoidApprox
		super(SpyALIFLayer, self).__init__(
			input_size=input_size,
			output_size=output_size,
			name=name,
			use_recurrent_connection=use_recurrent_connection,
			use_rec_eye_mask=use_rec_eye_mask,
			dt=dt,
			device=device,
			**kwargs
		)
		
		self.kappa = nn.Parameter(
			torch.tensor(self.kwargs["kappa"], dtype=torch.float32, device=self.device),
			requires_grad=self.kwargs["learn_kappa"]
		)
		self.rho = nn.Parameter(
			torch.tensor(np.exp(-dt / self.kwargs["tau_a"]), dtype=torch.float32, device=self.device),
			requires_grad=False
		)
	
	def _set_default_kwargs(self):
		self.kwargs.setdefault("tau_syn", 5.0 * self.dt)
		self.kwargs.setdefault("tau_mem", 10.0 * self.dt)
		self.kwargs.setdefault("tau_a", 200.0 * self.dt)
		self.kwargs.setdefault("threshold", 1.0)
		self.kwargs.setdefault("gamma", 100.0)
		self.kwargs.setdefault("kappa", 1.6)
		self.kwargs.setdefault("learn_kappa", False)
		self.kwargs.setdefault("spikes_regularization_factor", 0.0)
		self.kwargs.setdefault("hh_init", "zeros")
	
	def initialize_weights_(self):
		super().initialize_weights_()
		weight_scale = 0.2
		if "forward_weights" in self.kwargs:
			self.forward_weights.data = to_tensor(self.kwargs["forward_weights"]).to(self.device)
		else:
			torch.nn.init.normal_(self.forward_weights, mean=0.0, std=weight_scale / np.sqrt(int(self.input_size)))
		
		if "recurrent_weights" in self.kwargs and self.use_recurrent_connection:
			self.recurrent_weights.data = to_tensor(self.kwargs["recurrent_weights"]).to(self.device)
		elif self.use_recurrent_connection:
			torch.nn.init.normal_(self.recurrent_weights, mean=0.0, std=weight_scale / np.sqrt(int(self.output_size)))
	
	def create_empty_state(
			self,
			batch_size: int = 1,
			**kwargs
	) -> Tuple[torch.Tensor, ...]:
		"""
		Create an empty state in the following form:
			([membrane potential of shape (batch_size, self.output_size)],
			[synaptic current of shape (batch_size, self.output_size)],
			[spikes of shape (batch_size, self.output_size)])

		:param batch_size: The size of the current batch.
		:return: The current state.
		"""
		kwargs.setdefault("n_hh", 4)
		thr = self.threshold.detach().cpu().item()
		if self.kwargs["hh_init"] == "random":
			V_mu, V_std = self.kwargs.get("hh_init_mu", thr / 2.0), self.kwargs.get("hh_init_std", 0.341 * thr)
			gen = torch.Generator(device=self.device)
			gen.manual_seed(format_pseudo_rn_seed(self.kwargs.get("hh_init_seed", None)))
			V = torch.clamp_min(
				torch.rand(
					(batch_size, int(self.output_size)),
					device=self.device,
					dtype=torch.float32,
					requires_grad=True,
					generator=gen,
				) * V_std + V_mu, min=0.0
				)
			I = torch.rand(
				(batch_size, int(self.output_size)),
				device=self.device,
				dtype=torch.float32,
				requires_grad=True,
				generator=gen,
			)
			Z = self.spike_func.apply(V, self.threshold, self.gamma)
			V = V * (1.0 - Z)
			a = torch.rand(
				(batch_size, int(self.output_size)),
				device=self.device,
				dtype=torch.float32,
				requires_grad=True,
				generator=gen,
			)
			return tuple([V, I, a, Z])
		elif self.kwargs["hh_init"] == "inputs":
			# V_mu, V_std = self.kwargs.get("hh_init_mu", thr / 2.0), self.kwargs.get("hh_init_std", 0.341 * thr)
			V_mu, V_std = self.kwargs.get("hh_init_mu", 0.0), self.kwargs.get("hh_init_std", thr)
			gen = torch.Generator(device=self.device)
			gen.manual_seed(format_pseudo_rn_seed(self.kwargs.get("hh_init_seed", None)))
			I = torch.rand(
				(batch_size, int(self.output_size)),
				device=self.device,
				dtype=torch.float32,
				requires_grad=True,
				generator=gen,
			)
			Z = kwargs["inputs"].clone()
			V = (torch.rand(
				(batch_size, int(self.output_size)),
				device=self.device,
				dtype=torch.float32,
				requires_grad=True,
				generator=gen,
			) * V_std + V_mu)
			V = (self.beta * V + self.alpha * I) * (1.0 - Z)
			a = self.rho * torch.rand(
				(batch_size, int(self.output_size)),
				device=self.device,
				dtype=torch.float32,
				requires_grad=True,
				generator=gen,
			) * thr + Z
			return tuple([V, I, a, Z])
		return super(SpyLIFLayer, self).create_empty_state(batch_size=batch_size, **kwargs)
	
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
		self._regularization_l1 += self.kwargs["spikes_regularization_factor"] * torch.sum(next_Z)
		# self._n_spike_per_neuron += torch.sum(torch.sum(next_Z, dim=0), dim=0)
		# self._total_count += next_Z.shape[0]*next_Z.shape[1]
		# current_l2 = self.kwargs["spikes_regularization_factor"]*torch.sum(self._n_spike_per_neuron ** 2) / (self._total_count + 1e-6)
		# self._regularization_loss = self._regularization_l1 + current_l2
		self._regularization_loss = self._regularization_l1
		return self._regularization_loss
	
	def forward(
			self,
			inputs: torch.Tensor,
			state: Tuple[torch.Tensor, ...] = None,
			**kwargs
	):
		assert inputs.ndim == 2, f"Inputs must be of shape (batch_size, input_size), got {inputs.shape}."
		batch_size, nb_features = inputs.shape
		V, I_syn, a, Z = self._init_forward_state(state, batch_size, inputs=inputs)
		input_current = torch.matmul(inputs, self.forward_weights)
		if self.use_recurrent_connection:
			rec_current = torch.matmul(Z, torch.mul(self.recurrent_weights, self.rec_mask))
		else:
			rec_current = 0.0
		next_I_syn = self.alpha * I_syn + input_current + rec_current
		next_V = (self.beta * V + next_I_syn) * (1.0 - Z.detach())
		next_a = self.rho * a + Z  # a^{t+1} = \rho * a_j^t + z_j^t
		A = self.threshold + self.kappa * next_a  # A_j^t = v_{th} + \kappa * a_j^t
		next_Z = self.spike_func.apply(next_V, A, self.gamma)  # z_j^t = H(v_j^t - A_j^t)
		return next_Z, (next_V, next_I_syn, next_a, next_Z)


# @inherit_fields_docstring(fields=["Attributes"], bases=[LIFLayer])
class ALIFLayer(LIFLayer):
	"""
	The ALIF dynamic, inspired by Bellec and \\textit{al.} :cite:t:`bellec_solution_2020`, is very
	similar to the LIF dynamics (class :class:`LIFLayer`). In fact, ALIF has exactly the same potential
	update equation as LIF. The difference comes
	from the fact that the threshold potential varies with time and neuron input. Indeed, the threshold
	is increased at each output pulse and is then decreased with a certain rate in order to come back to
	its starting threshold :math:`V_{\\text{th}}`. The threshold equation from :class:`LIFLayer` is thus slightly
	modified by changing :math:`V_{\\text{th}} \\to A_j^t`. Thus, the output of neuron :math:`j` at time :math:`t`
	denoted :math:`z_j^t` is redefined by the equation :eq:`alif_z`.
	
	.. math::
		:label: alif_z
		
		\\begin{equation}
			z_j^t = H(V_j^t - A_j^t)
		\\end{equation}
	
	The update of the activation threshold is then described by :eq:`alif_A`.
	
	.. math::
		:label: alif_A
	
		\\begin{equation}
			A_j^t = V_{\\text{th}} + \\beta a_j^t
		\\end{equation}
	
	with the adaptation variable :math:`a_j^t` described by :eq:`alif_a` and :math:`\\beta` an amplification
	factor greater than 1 and typically equivalent to :math:`\\beta\\approx 1.6` :cite:t:`bellec_solution_2020`.
	
	.. math::
		:label: alif_a
	
		\\begin{equation}
			a_j^{t+1} = \\rho a_j + z_j^t
		\\end{equation}
	
	With the decay factor :math:`\\rho` as:
	
	.. math::
		:label: alif_rho
		
		\\begin{equation}
			\\rho = e^{-\\frac{\\Delta t}{\\tau_a}}
		\\end{equation}
		
	:Attributes:
		- :attr:`beta`: The amplification factor of the threshold potential :math:`\\beta`.
		- :attr:`rho`: The decay factor of the adaptation variable :math:`\\rho`.
		
	"""
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			use_recurrent_connection: bool = True,
			use_rec_eye_mask: bool = False,
			spike_func: Type[SpikeFunction] = HeavisideSigmoidApprox,
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

	def create_empty_state(
			self,
			batch_size: int = 1,
			**kwargs
	) -> Tuple[torch.Tensor, ...]:
		"""
		Create an empty state in the following form:
			[[membrane potential of shape (batch_size, self.output_size)]
			[current threshold of shape (batch_size, self.output_size)]
			[spikes of shape (batch_size, self.output_size)]]
		
		:param batch_size: The size of the current batch.
		:type batch_size: int
		
		:return: The current state.
		:rtype: Tuple[torch.Tensor, ...]
		"""
		kwargs.setdefault("n_hh", 3)
		return super(ALIFLayer, self).create_empty_state(batch_size=batch_size, **kwargs)

	def forward(
			self,
			inputs: torch.Tensor,
			state: Tuple[torch.Tensor, ...] = None,
			**kwargs
	):
		assert inputs.ndim == 2
		batch_size, nb_features = inputs.shape
		V, a, Z = self._init_forward_state(state, batch_size, inputs=inputs)
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
		:type state: Optional[Any]
		
		:return: The updated regularization loss.
		:rtype: torch.Tensor
		"""
		next_V, next_a, next_Z = state
		self._regularization_loss += self.kwargs["spikes_regularization_factor"]*torch.sum(next_Z)
		# self._regularization_loss += 2e-6*torch.mean(torch.sum(next_Z, dim=-1)**2)
		return self._regularization_loss
	

class BellecLIFLayer(LIFLayer):
	"""
	Layer implementing the LIF neuron model from the paper:
		"A solution to the learning dilemma for recurrent networks of spiking neurons"
		by Bellec et al. (2020) :cite:t:`bellec_solution_2020`.
	"""
	
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			use_recurrent_connection: bool = True,
			use_rec_eye_mask: bool = True,
			spike_func: Type[SpikeFunction] = HeavisidePhiApprox,
			dt: float = 1e-3,
			device: Optional[torch.device] = None,
			**kwargs
	):
		super().__init__(
			input_size=input_size,
			output_size=output_size,
			name=name,
			use_recurrent_connection=use_recurrent_connection,
			use_rec_eye_mask=use_rec_eye_mask,
			spike_func=spike_func,
			dt=dt,
			device=device,
			**kwargs
		)
		
	def forward(
			self,
			inputs: torch.Tensor,
			state: Tuple[torch.Tensor, ...] = None,
			**kwargs
	):
		assert inputs.ndim == 2
		batch_size, nb_features = inputs.shape
		V, Z = self._init_forward_state(state, batch_size, inputs=inputs)
		input_current = torch.matmul(inputs, self.forward_weights)
		if self.use_recurrent_connection:
			rec_current = torch.matmul(Z, torch.mul(self.recurrent_weights, self.rec_mask))
		else:
			rec_current = 0.0
		next_V = (self.alpha * V + input_current + rec_current) - Z.detach()*self.threshold
		next_Z = self.spike_func.apply(next_V, self.threshold, self.gamma)
		return next_Z, (next_V, next_Z)


# @inherit_fields_docstring(fields=["Attributes"], bases=[BaseNeuronsLayer])
class IzhikevichLayer(BaseNeuronsLayer):
	"""
	Izhikevich p.274
	
	Not usable for now, stay tuned.
	"""
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			use_recurrent_connection=True,
			use_rec_eye_mask=True,
			spike_func: Type[SpikeFunction] = HeavisideSigmoidApprox,
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
		super().initialize_weights_()
		gain = 1.0
		for param in self.parameters():
			if param.ndim > 2:
				torch.nn.init.xavier_normal_(param, gain=gain)
			else:
				torch.nn.init.normal_(param, std=gain)

	def create_empty_state(
			self,
			batch_size: int = 1,
			**kwargs
	) -> Tuple[torch.Tensor, ...]:
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

	def forward(
			self,
			inputs: torch.Tensor,
			state: Tuple[torch.Tensor, ...] = None,
			**kwargs
	):
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


# @inherit_fields_docstring(fields=["Attributes"], bases=[BaseNeuronsLayer])
class WilsonCowanLayer(BaseNeuronsLayer):
	"""
	This layer is use for Wilson-Cowan neuronal dynamics.
	This dynamic is also referred to as firing rate model.
	Wilson-Cowan dynamic is great for neuronal calcium activity.
	This layer use recurrent neural network (RNN).
	The number of parameters that are trained is N^2 (+2N if mu and r is train)
	where N is the number of neurons.
	
	For references, please read:
		
		- Excitatory and Inhibitory Interactions in Localized Populations of Model Neurons :cite:t:`wilson1972excitatory`
		- Beyond Wilson-Cowan dynamics: oscillations and chaos without inhibitions :cite:t:`PainchaudDoyonDesrosiers2022`
		- Neural Network dynamic :cite:t:`VogelsTimRajanAbbott2005NeuralNetworkDynamics`.

	The Wilson-Cowan dynamic is one of many dynamical models that can be used
	to model neuronal activity. To explore more continuous and Non-linear dynamics,
	please read Nonlinear Neural Network: Principles, Mechanisms, and Architecture :cite:t:`GROSSBERG198817`.
	

	"""
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			dt: float = 1e-3,
			use_recurrent_connection: bool = False,
			device=None,
			**kwargs
	):
		"""
		:param input_size: size of the input
		:type input_size: Optional[SizeTypes]
		:param output_size: size of the output
			If we are predicting time series -> input_size = output_size
		:type output_size: Optional[SizeTypes]
		:param learning_type: Type of learning for the gradient descent
		:type learning_type: LearningType
		:param dt: Time step (Euler's discretisation)
		:type dt: float
		:param device: device for computation
		:type device: torch.device
		:param kwargs: Additional parameters for the Wilson-Cowan dynamic.
		
		:keyword Union[torch.Tensor, np.ndarray] forward_weights: Forward weights of the layer.
		:keyword float std_weight: Instability of the initial random matrix.
		:keyword Union[float, torch.Tensor] mu: Activation threshold. If torch.Tensor -> shape (1, number of neurons).
		:keyword float mean_mu: Mean of the activation threshold (if learn_mu is True).
		:keyword float std_mu: Standard deviation of the activation threshold (if learn_mu is True).
		:keyword bool learn_mu: Whether to train the activation threshold.
		:keyword float tau: Decay constant of RNN unit.
		:keyword bool learn_tau: Whether to train the decay constant.
		:keyword float r: Transition rate of the RNN unit. If torch.Tensor -> shape (1, number of neurons).
		:keyword float mean_r: Mean of the transition rate (if learn_r is True).
		:keyword float std_r: Standard deviation of the transition rate (if learn_r is True).
		:keyword bool learn_r: Whether to train the transition rate.

		Remarks: Parameter mu and r can only be a parameter as a vector.
		"""
		super(WilsonCowanLayer, self).__init__(
			input_size=input_size,
			output_size=output_size,
			use_recurrent_connection=use_recurrent_connection,
			dt=dt,
			device=device,
			**kwargs
		)
		self.std_weight = self.kwargs["std_weight"]
		self.mu = torch.nn.Parameter(to_tensor(self.kwargs["mu"]).to(self.device), requires_grad=False)
		self.mean_mu = self.kwargs["mean_mu"]
		self.std_mu = self.kwargs["std_mu"]
		self.learn_mu = self.kwargs["learn_mu"]
		self.tau = torch.nn.Parameter(to_tensor(self.kwargs["tau"]).to(self.device), requires_grad=False)
		self.learn_tau = self.kwargs["learn_tau"]
		self.r_sqrt = torch.nn.Parameter(
			torch.sqrt(to_tensor(self.kwargs["r"], dtype=torch.float32)).to(self.device), requires_grad=False
		)
		self.mean_r = self.kwargs["mean_r"]
		self.std_r = self.kwargs["std_r"]
		self.learn_r = self.kwargs["learn_r"]
		self.activation = self._init_activation(self.kwargs["activation"])
		
	def _init_activation(self, activation: Union[torch.nn.Module, str]):
		"""
		Initialise the activation function.
		
		:param activation: Activation function.
		:type activation: Union[torch.nn.Module, str]
		"""
		str_to_activation = {
			"identity": torch.nn.Identity(),
			"relu": torch.nn.ReLU(),
			"tanh": torch.nn.Tanh(),
			"sigmoid": torch.nn.Sigmoid(),
		}
		if isinstance(activation, str):
			assert activation in str_to_activation.keys(), f"Activation {activation} is not implemented."
			self.activation = str_to_activation[activation]
		else:
			self.activation = activation
		return self.activation

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
		self.kwargs.setdefault("activation", "sigmoid")

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
		super().initialize_weights_()
		if "forward_weights" in self.kwargs:
			self.forward_weights = to_tensor(self.kwargs["forward_weights"]).to(self.device)
		else:
			torch.nn.init.normal_(self._forward_weights, mean=0.0, std=self.std_weight)

		# If mu is not a parameter, it takes the value 0.0 unless stated otherwise by user
		# If mu is a parameter, it is initialized as a vector with the correct mean and std
		# unless stated otherwise by user.
		if self.learn_mu:
			if self.mu.dim() == 0:  # if mu is a scalar and a parameter -> convert it to a vector
				self.mu.data = torch.empty((1, int(self.output_size)), dtype=torch.float32, device=self.device)
			self.mu = torch.nn.Parameter(self.mu, requires_grad=self.requires_grad)
			torch.nn.init.normal_(self.mu, mean=self.mean_mu, std=self.std_mu)
		if self.learn_r:
			_r = torch.empty((1, int(self.output_size)), dtype=torch.float32, device=self.device)
			torch.nn.init.normal_(_r, mean=self.mean_r, std=self.std_r)
			self.r_sqrt = torch.nn.Parameter(torch.sqrt(torch.abs(_r)), requires_grad=self.requires_grad)
		if self.learn_tau:
			self.tau = torch.nn.Parameter(self.tau, requires_grad=self.requires_grad)

	def create_empty_state(self, batch_size: int = 1, **kwargs) -> Tuple[torch.Tensor]:
		if self.kwargs["hh_init"] == "zeros":
			state = [torch.zeros(
				(batch_size, int(self.output_size)),
				device=self.device,
				dtype=torch.float32,
				requires_grad=True,
			) for _ in range(1)]
		elif self.kwargs["hh_init"] == "random":
			mu, std = self.kwargs.get("hh_init_mu", 0.0), self.kwargs.get("hh_init_std", 1.0)
			gen = torch.Generator(device=self.device)
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
			state: Optional[Tuple[torch.Tensor, ...]] = None,
			**kwargs
	) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
		"""
		Forward pass.
		With Euler discretisation, Wilson-Cowan equation becomes:
		
		output = input * (1 - dt/tau) + dt/tau * (1 - input @ r) * sigmoid(input @ forward_weight - mu)
		
		:param inputs: time series at a time t of shape (batch_size, number of neurons)
			Remark: if you use to compute a time series, use batch_size = 1.
		:type inputs: torch.Tensor
		:param state: State of the layer (only for SNN -> not use for RNN)
		:type state: Optional[Tuple[torch.Tensor, ...]]
		
		:return: (time series at a time t+1, State of the layer -> None)
		:rtype: Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]
		"""
		batch_size, nb_features = inputs.shape
		hh, = self._init_forward_state(state, batch_size, inputs=inputs)
		ratio_dt_tau = self.dt / self.tau

		if self.use_recurrent_connection:
			rec_inputs = torch.matmul(hh, torch.mul(self.recurrent_weights, self.rec_mask))
		else:
			rec_inputs = 0.0

		transition_rate = (1 - hh * self.r)
		activation = self.activation(rec_inputs + torch.matmul(inputs, self.forward_weights) - self.mu)
		output = hh * (1 - ratio_dt_tau) + transition_rate * activation * ratio_dt_tau
		return output, (output, )


class WilsonCowanCURBDLayer(WilsonCowanLayer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def forward(
			self,
			inputs: torch.Tensor,
			state: Optional[Tuple[torch.Tensor, ...]] = None,
			**kwargs
	) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
		batch_size, nb_features = inputs.shape
		hh, = self._init_forward_state(state, batch_size, inputs=inputs)
		output = self.activation(hh)
		
		if self.use_recurrent_connection:
			rec_inputs = torch.matmul(hh, torch.mul(self.recurrent_weights, self.rec_mask))
		else:
			rec_inputs = 0.0
		
		r = rec_inputs + torch.matmul(output, self.forward_weights)
		hh = hh + self.dt * (r - hh) / self.tau
		return output, (hh, )


# @inherit_fields_docstring(fields=["Attributes"], bases=[BaseNeuronsLayer])
class LILayer(BaseNeuronsLayer):
	"""
	The integration in time of these dynamics is done using the equation
	:eq:`li_v` inspired by Bellec and al. :cite:t:`bellec_solution_2020`.
	
	.. math::
		:label: li_v
		
		\\begin{equation}
			V_j^{t+\\Delta t} = \\kappa V_j^{t} + \\sum_{i}^N W_{ij}x_i^{t+\\Delta t} + b_j
		\\end{equation}
	
	.. math::
		:label: li_kappa
		
		\\begin{equation}
			\\kappa = e^{-\\frac{\\Delta t}{\\tau_{\\text{mem}}}}
		\\end{equation}
	
	
	The parameters of the equation :eq:`li_v` are:
		
		- :math:`N` is the number of neurons in the layer.
		- :math:`V_j^t` is the synaptic potential of the neuron :math:`j` at time :math:`t`.
		- :math:`\\Delta t` is the integration time step.
		- :math:`\\kappa` is the decay constant of the synaptic current over time (equation :eq:`li_kappa`).
		- :math:`W_{ij}^{\\text{rec}}` is the recurrent weight of the neuron :math:`i` to the neuron :math:`j`.
		- :math:`W_{ij}^{\\text{in}}` is the input weight of the neuron :math:`i` to the neuron :math:`j`.
		- :math:`x_i^{t}` is the input of the neuron :math:`i` at time :math:`t`.
	
	:Attributes:
		- :attr:`bias_weights` (torch.nn.Parameter): Bias weights of the layer.
		- :attr:`kappa` (torch.nn.Parameter): Decay constant of the synaptic current over time see equation :eq:`li_kappa`.
	
	"""
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			dt: float = 1e-3,
			device: Optional[torch.device] = None,
			**kwargs
	):
		super(LILayer, self).__init__(
			input_size=input_size,
			output_size=output_size,
			name=name,
			use_recurrent_connection=False,
			dt=dt,
			device=device,
			**kwargs
		)
		self.bias_weights = None
		self.kappa = torch.nn.Parameter(
			torch.tensor(self.kwargs["kappa"], dtype=torch.float32, device=self.device),
			requires_grad=self.kwargs["learn_kappa"]
		)

	def _set_default_kwargs(self):
		self.kwargs.setdefault("tau_out", 10.0 * self.dt)
		self.kwargs.setdefault("kappa", np.exp(-self.dt / self.kwargs["tau_out"]))
		self.kwargs.setdefault("learn_kappa", False)
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

	def create_empty_state(
			self,
			batch_size: int = 1,
			**kwargs
	) -> Tuple[torch.Tensor, ...]:
		"""
		Create an empty state in the following form:
			[membrane potential of shape (batch_size, self.output_size)]
		
		:param batch_size: The size of the current batch.
		:return: The current state.
		"""
		kwargs.setdefault("n_hh", 1)
		return super(LILayer, self).create_empty_state(batch_size=batch_size, **kwargs)

	def forward(
			self,
			inputs: torch.Tensor,
			state: Tuple[torch.Tensor, ...] = None,
			**kwargs
	):
		assert inputs.ndim == 2
		batch_size, nb_features = inputs.shape
		# state = self._init_forward_state(state, batch_size)
		V, = self._init_forward_state(state, batch_size, inputs=inputs)
		next_V = self.kappa * V + torch.matmul(inputs, self.forward_weights) + self.bias_weights
		return next_V, (next_V, )


# @inherit_fields_docstring(fields=["Attributes"], bases=[BaseNeuronsLayer])
class SpyLILayer(BaseNeuronsLayer):
	"""
	The SpyLI dynamics is a more complex variant of the LI dynamics (class :class:`LILayer`) allowing it to have a
	greater power of expression. This variant is also inspired by Neftci :cite:t:`neftci_surrogate_2019` and also
	contains  two differential equations like the SpyLIF dynamics :class:`SpyLIFLayer`. The equation :eq:`SpyLI_I`
	presents the synaptic current update equation with euler integration while the equation :eq:`SpyLI_V` presents the
	synaptic potential update.

	.. math::
		:label: SpyLI_I

		\\begin{equation}
			I_{\\text{syn}, j}^{t+\\Delta t} = \\alpha I_{\text{syn}, j}^{t} +
			\\sum_{i}^{N} W_{ij}^{\\text{rec}} I_{\\text{syn}, j}^{t}
			+ \\sum_i^{N} W_{ij}^{\\text{in}} x_i^{t+\\Delta t}
		\\end{equation}


	.. math::
		:label: SpyLI_V

		\\begin{equation}
			V_j^{t+\\Delta t} = \\beta V_j^t + I_{\\text{syn}, j}^{t+\\Delta t} + b_j
		\\end{equation}


	.. math::
		:label: spyli_alpha

		\\begin{equation}
			\\alpha = e^{-\\frac{\\Delta t}{\\tau_{\\text{syn}}}}
		\\end{equation}

	with :math:`\\tau_{\\text{syn}}` being the decay time constant of the synaptic current.

	.. math::
		:label: spyli_beta

		\\begin{equation}
			\\beta = e^{-\\frac{\\Delta t}{\\tau_{\\text{mem}}}}
		\\end{equation}

	with :math:`\\tau_{\\text{syn}}` being the decay time constant of the synaptic current.

	SpyTorch library: https://github.com/surrogate-gradient-learning/spytorch.

	The variables of the equations :eq:`SpyLI_I` and :eq:`SpyLI_V` are described by the following definitions:

		- :math:`N` is the number of neurons in the layer.
		- :math:`I_{\\text{syn}, j}^{t}` is the synaptic current of neuron :math:`j` at time :math:`t`.
		- :math:`V_j^t` is the synaptic potential of the neuron :math:`j` at time :math:`t`.
		- :math:`\\Delta t` is the integration time step.
		- :math:`\\alpha` is the decay constant of the synaptic current over time (equation :eq:`spyli_alpha`).
		- :math:`\\beta` is the decay constant of the membrane potential over time (equation :eq:`spyli_beta`).
		- :math:`W_{ij}^{\\text{rec}}` is the recurrent weight of the neuron :math:`i` to the neuron :math:`j`.
		- :math:`W_{ij}^{\\text{in}}` is the input weight of the neuron :math:`i` to the neuron :math:`j`.
		- :math:`x_i^{t}` is the input of the neuron :math:`i` at time :math:`t`.

	:Attributes:
		- :attr:`alpha` (torch.nn.Parameter): Decay constant of the synaptic current over time (equation :eq:`spyli_alpha`).
		- :attr:`beta` (torch.nn.Parameter): Decay constant of the membrane potential over time (equation :eq:`spyli_beta`).
		- :attr:`gamma` (torch.nn.Parameter): Slope of the Heaviside function (:math:`\\gamma`).
	"""
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			dt: float = 1e-3,
			device: Optional[torch.device] = None,
			**kwargs
	):
		super(SpyLILayer, self).__init__(
			input_size=input_size,
			output_size=output_size,
			name=name,
			use_recurrent_connection=False,
			dt=dt,
			device=device,
			**kwargs
		)
		self.bias_weights = None
		self.alpha = torch.nn.Parameter(
			torch.tensor(self.kwargs["alpha"], dtype=torch.float32, device=self.device),
			requires_grad=self.kwargs["learn_alpha"]
		)
		self.beta = torch.nn.Parameter(
			torch.tensor(self.kwargs["beta"], dtype=torch.float32, device=self.device),
			requires_grad=self.kwargs["learn_beta"]
		)

	def _set_default_kwargs(self):
		self.kwargs.setdefault("tau_syn", 5.0 * self.dt)
		self.kwargs.setdefault("alpha", np.exp(-self.dt / self.kwargs["tau_syn"]))
		self.kwargs.setdefault("learn_alpha", False)
		self.kwargs.setdefault("tau_mem", 10.0 * self.dt)
		self.kwargs.setdefault("beta", np.exp(-self.dt / self.kwargs["tau_mem"]))
		self.kwargs.setdefault("learn_beta", False)
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

	def create_empty_state(
			self,
			batch_size: int = 1,
			**kwargs
	) -> Tuple[torch.Tensor, ...]:
		"""
		Create an empty state in the following form:
			[membrane potential of shape (batch_size, self.output_size),
			synaptic current of shape (batch_size, self.output_size)]
		
		:param batch_size: The size of the current batch.
		:return: The current state.
		"""
		kwargs.setdefault("n_hh", 2)
		return super(SpyLILayer, self).create_empty_state(batch_size=batch_size, **kwargs)

	def forward(
			self,
			inputs: torch.Tensor,
			state: Tuple[torch.Tensor, ...] = None,
			**kwargs
	):
		assert inputs.ndim == 2
		batch_size, nb_features = inputs.shape
		V, I_syn = self._init_forward_state(state, batch_size, inputs=inputs)
		next_I_syn = self.alpha * I_syn + torch.matmul(inputs, self.forward_weights)
		next_V = self.beta * V + next_I_syn + self.bias_weights
		return next_V, (next_V, next_I_syn)


LayerType2Layer = {
	LayerType.LIF: LIFLayer,
	LayerType.ALIF: ALIFLayer,
	LayerType.Izhikevich: IzhikevichLayer,
	LayerType.LI: LILayer,
	LayerType.SpyLIF: SpyLIFLayer,
	LayerType.SpyALIF: SpyALIFLayer,
	LayerType.SpyLI: SpyLILayer,
}

