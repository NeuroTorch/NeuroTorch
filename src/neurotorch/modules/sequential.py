import warnings
from collections import defaultdict, OrderedDict
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import (
	BaseLayer,
	BaseModel,
	LIFLayer,
	LayerType,
	LayerType2Layer,
	SpikeFuncType,
	SpikeFuncType2Func,
	SpikeFunction
)
from ..dimension import Dimension

Acceptable_Spike_Func = Union[Type[SpikeFunction], SpikeFuncType]
Acceptable_Spike_Funcs = Union[Acceptable_Spike_Func, Iterable[Acceptable_Spike_Func]]
Acceptable_Layer_Type = Union[Type[BaseLayer], LayerType]
Acceptable_Layer_Types = Union[Acceptable_Layer_Type, Iterable[Acceptable_Layer_Type]]
IntDimension = Union[int, Dimension]


class SequentialModel(BaseModel):

	@staticmethod
	def _format_hidden_outputs_traces(
			hidden_states: Dict[str, List[Tuple[torch.Tensor, ...]]]
	) -> Dict[str, Tuple[torch.Tensor, ...]]:
		"""
		Permute the hidden states to have a dictionary of shape {layer_name: (tensor, ...)}
		trace can be a list of :
			- Tensor -> list[torch.Tensor]
			- Tuple or list of Tensor or None -> Iterable[torch.Tensor] or Iterable[None]
		If the list has those format, it will be converted to a dictionary of shape {layer_name: (tensor, ...)}
		However, if you decide to format trace differently (empty list, numpy array ...) it won't be reshape into
		a dict. The new hidden states will therefore stay the same as the hidden_state.
		Also, please note that if all the element of your list are not the same type, it will raise an error. However,
		if you use a list of iterable, it will NOT check if all the element of the iterable are the same type. This
		decision was done to reduce the computation time. Make sure all the element of your list are the same type to
		avoid error.
		:param hidden_states: Dictionary of hidden states
		:return: Dictionary of hidden states with the shape {layer_name: (tensor, ...)}
		"""
		new_hidden_states = {}
		for layer_name, trace in hidden_states.items():
			if len(trace) == 0:
				new_hidden_states[layer_name] = trace
				continue
			trace_element_type = type(trace[0])
			if not all(isinstance(e, trace_element_type) for e in trace):
				raise ValueError("The hidden states returned by the layers must always have the same type")
			# if trace is a list of tensors :
			if issubclass(trace_element_type, torch.Tensor):
				new_hidden_states[layer_name] = torch.stack(trace, dim=1)

			# if trace is a list of iterable: :
			elif issubclass(trace_element_type, Iterable):
				internal_trace_element_type = type(trace[0][0])
				# if the iterable is a list of None:
				if issubclass(internal_trace_element_type, type(None)):
					new_hidden_states[layer_name] = [None] * len(trace)
				# if the iterable is a list of tensors:
				elif issubclass(internal_trace_element_type, torch.Tensor):
					new_hidden_states[layer_name] = tuple([torch.stack(e, dim=1) for e in list(zip(*trace))])
				# If the iterable has an other format, it will be kept as it is
				else:
					new_hidden_states[layer_name] = trace
			# If the list has another format, it will be kept as it is
			else:
				new_hidden_states[layer_name] = trace
			# else (if trace is a list of scalar or a list of None):
			# new_hidden_states[layer_name] = trace
			#
		return new_hidden_states

	@staticmethod
	def _format_input_output_layers(
			layers: Iterable[Union[Iterable[BaseLayer], BaseLayer]],
			default_prefix_layer_name: str = "layer",
	) -> OrderedDict[str, BaseLayer]:
		"""
		Format the input or output layers. The format is an ordered dictionary of the form {layer_name: layer}.
		:param layers: The input or output layers.
		:param default_prefix_layer_name: The default prefix of the layer name. The prefix is used when the name of
		the layer is not specified.
		:return: The formatted input or output layers.
		"""
		layers: Iterable[BaseLayer] = [layers] if not isinstance(layers, Iterable) else layers
		if isinstance(layers, Mapping):
			all_base_layer = all(isinstance(layer, (BaseLayer, dict)) for _, layer in layers.items())

			for layer_key, layer in layers.items():
				if not layer.name_is_set:
					layer.name = layer_key
			assert all(layer_key == layer.name for layer_key, layer in layers.items()), \
				"The layer names must be the same as the keys."
		else:
			all_base_layer = all(isinstance(layer, (BaseLayer, dict)) for layer in layers)
		assert all_base_layer, "All layers must be of type BaseLayer"
		if not isinstance(layers, dict):
			for layer_idx, layer in enumerate(layers):
				if not layer.name_is_set:
					layer.name = f"{default_prefix_layer_name}_{layer_idx}"
			assert len([layer.name for layer in layers]) == len(set([layer.name for layer in layers])), \
				"There are layers with the same name."
			layers = OrderedDict((layer.name, layer) for layer in layers)
		return layers

	@staticmethod
	def _format_hidden_layers(
			layers: Iterable[BaseLayer],
			default_prefix_layer_name: str = "hidden",
	) -> List[BaseLayer]:
		"""
		Format the hidden layers. The format is a list of the form [layer, ...].
		:param layers: The hidden layers.
		:param default_prefix_layer_name: The default prefix of the layer name. The prefix is used when the name of
		the layer is not specified.
		:return: The formatted hidden layers.
		"""
		assert all([isinstance(layer, BaseLayer) for layer in layers]), \
			"All hidden layers must be of type BaseLayer"
		for i, layer in enumerate(layers):
			if not layer.name_is_set:
				layer.name = f"{default_prefix_layer_name}_{i}"
		return list(layers)

	@staticmethod
	def _format_layers(
			layers: Iterable[Union[Iterable[BaseLayer], BaseLayer]]
	) -> Tuple[OrderedDict, List, OrderedDict]:
		"""
		Format the given layers. The format is a tuple of the form
		(
			OrderedDict({input_layer_name: input_layer}),
			List(hidden_layers),
			OrderedDict({output_layer_name: output_layer}),
		).
		:param layers: The layers.
		:return: The formatted layers.
		"""
		if not isinstance(layers, Iterable):
			layers = [layers]
		if len(layers) > 1:
			input_layers = SequentialModel._format_input_output_layers(layers[0], "input")
		else:
			input_layers = nn.ModuleDict()

		if len(layers) > 2:
			hidden_layers = layers[1:-1]
			if not isinstance(hidden_layers, Iterable):
				hidden_layers = [hidden_layers]
		else:
			hidden_layers = []
		hidden_layers = SequentialModel._format_hidden_layers(hidden_layers)

		output_layers = SequentialModel._format_input_output_layers(layers[-1], "output")
		return input_layers, hidden_layers, output_layers

	@staticmethod
	def _layers_containers_to_modules(
			inputs_layers: OrderedDict,
			hidden_layers: List,
			outputs_layers: OrderedDict
	) -> Tuple[nn.ModuleDict, nn.ModuleList, nn.ModuleDict]:
		"""
		Convert the input, hidden and output layers containers to modules.
		:param inputs_layers: The input layers.
		:param hidden_layers: The hidden layers.
		:param outputs_layers: The output layers.
		:return: The input, hidden and output layers modules.
		"""
		input_layers = nn.ModuleDict(inputs_layers)
		hidden_layers = nn.ModuleList(hidden_layers)
		output_layers = nn.ModuleDict(outputs_layers)
		return input_layers, hidden_layers, output_layers

	@staticmethod
	def _format_layer_type_(
			layer_type: Optional[Acceptable_Layer_Type]
	) -> Optional[Type[BaseLayer]]:
		warnings.warn(
			"This function is not used anymore in the SequentialModel.",
			DeprecationWarning
		)
		if isinstance(layer_type, LayerType):
			layer_type = LayerType2Layer[layer_type]
		return layer_type

	@staticmethod
	def _format_hidden_neurons_(n_hidden_neurons: Optional[Union[int, Iterable[int]]]) -> List[int]:
		warnings.warn(
			"This function is not used anymore in the SequentialModel.",
			DeprecationWarning
		)
		if n_hidden_neurons is None:
			return []
		if not isinstance(n_hidden_neurons, Iterable):
			n_hidden_neurons = [n_hidden_neurons]
		return n_hidden_neurons

	def __new__(
			cls,
			*,
			input_sizes: Optional[Union[Dict[str, IntDimension], List[int], IntDimension]] = None,
			output_sizes: Optional[Union[Dict[str, int], List[int], int]] = None,
			# n_hidden_neurons: Optional[Union[int, Iterable[int]]] = None,
			use_recurrent_connection: Optional[Union[bool, Iterable[bool]]] = None,
			spike_funcs: Optional[Acceptable_Spike_Funcs] = None,
			hidden_layer_types: Optional[Acceptable_Layer_Types] = None,
			readout_layer_type: Optional[Acceptable_Layer_Type] = None,
			layers: Optional[Iterable[BaseLayer]] = None,
			**kwargs
	):
		auto_construct_parameters = {
			"input_sizes": input_sizes,
			"output_sizes": output_sizes,
			# n_hidden_neurons,
			"use_recurrent_connection": use_recurrent_connection,
			"spike_funcs": spike_funcs,
			"hidden_layer_types": hidden_layer_types,
			"readout_layer_type": readout_layer_type
		}
		if layers is not None:
			assert all([p is None for _, p in auto_construct_parameters.items()]), \
				f"You can't use the named parameters: " \
				f"{auto_construct_parameters} and layers at the same time"
			return super(SequentialModel, cls).__new__(cls)
		raise NotImplementedError("Auto construct feature is not available yet.")

	def __init__(
			self,
			layers: Iterable[Union[Iterable[BaseLayer], BaseLayer]],
			foresight_time_steps: int = 0,
			name: str = "snn",
			checkpoint_folder: str = "checkpoints",
			device: Optional[torch.device] = None,
			input_transform: Optional[Union[Dict[str, Callable], List[Callable]]] = None,
			output_transform: Optional[Union[Dict[str, Callable], List[Callable]]] = None,
			**kwargs
	):
		"""
		The SequentialModel is a neural network that is constructed by stacking layers.
		
		:param layers: The layers to be used in the model. The following structure is expected:
			layers = [
				[*inputs_layers, ],
				*hidden_layers,
				[*output_layers, ]
			]
			or
			layers = [
				input_layer,
				*hidden_layers,
				output_layer
			].
		:param foresight_time_steps: The number of time steps to predict in the future. When multiple inputs or outputs
			are given, the outputs of the network are given to the inputs in the same order as they were specified in
			the construction of the network. In other words, the first output is given to the first input, the second
			output is given to the second input, and so on. If there are fewer outputs than inputs, the last inputs are
			not considered as recurrent inputs, so they are not fed.
		:param name: The name of the model.
		:param checkpoint_folder: The folder where the checkpoints are saved.
		:param device: The device to use.
		:param input_transform: The transform to apply to the input. The input_transform must work on a single datum.
		:param output_transform: The transform to apply to the output trace. The output_transform must work batch-wise.
		:param kwargs:
				memory_size (Optional[int]): The size of the memory buffer. The output of each layer is stored in
					the memory buffer. If the memory_size is not specified, the memory_size is set to the number
					of time steps of the inputs.
		"""
		input_layers, hidden_layers, output_layers = self._format_layers(layers)
		self._ordered_inputs_names = [layer.name for _, layer in input_layers.items()]
		self._ordered_outputs_names = [layer.name for _, layer in output_layers.items()]
		super(SequentialModel, self).__init__(
			input_sizes={layer.name: layer.input_size for _, layer in input_layers.items()},
			output_size={layer.name: layer.output_size for _, layer in output_layers.items()},
			name=name,
			checkpoint_folder=checkpoint_folder,
			device=device,
			input_transform=input_transform,
			output_transform=output_transform,
			**kwargs
		)
		self._default_n_hidden_neurons = self.kwargs.get("n_hidden_neurons", 128)
		self.input_layers, self.hidden_layers, self.output_layers = self._layers_containers_to_modules(
			input_layers, hidden_layers, output_layers
		)
		assert len(self.get_all_layers_names()) == len(set(self.get_all_layers_names())), \
			"There are layers with the same name."
		self.foresight_time_steps = foresight_time_steps
		# self.n_hidden_neurons = self._format_hidden_neurons_(n_hidden_neurons)
		# self.spike_func = self._format_spike_funcs_(spike_funcs)
		# self.hidden_layer_types: List[Type] = self._format_layer_types_(hidden_layer_types)
		# self.readout_layer_type = self._format_layer_type_(readout_layer_type)  # TODO: change for multiple readout layers
		# self._add_layers_()
		self._memory_size: Optional[int] = self.kwargs.get("memory_size", None)
		assert self._memory_size is None or self._memory_size > 0, "The memory size must be greater than 0 or None."
		self._outputs_to_inputs_names_map: Optional[Dict[str, str]] = None

	@BaseModel.device.setter
	def device(self, device: torch.device):
		"""
		Set the device of the model and all its layers.
		:param device: The device to use.
		:return: None
		"""
		BaseModel.device.fset(self, device)
		for layer in self.get_all_layers():
			layer.device = device

	def get_all_layers(self) -> List[nn.Module]:
		"""
		Get all the layers of the model as a list. The order of the layers is the same as the order of the layers in the
		model.
		:return: A list of all the layers of the model.
		"""
		return list(self.input_layers.values()) + list(self.hidden_layers) + list(self.output_layers.values())

	def get_all_layers_names(self) -> List[str]:
		"""
		Get all the names of the layers of the model. The order of the layers is the same as the order of the layers in
		the model.
		:return: A list of all the names of the layers of the model.
		"""
		return [layer.name for layer in self.get_all_layers()]

	def get_dict_of_layers(self):
		return {layer.name: layer for layer in self.get_all_layers()}

	def get_layer(self, name: Optional[str] = None) -> nn.Module:
		"""
		Get a layer of the model. If the name is None, the first layer is returned which is useful when the model has
		only one layer.
		:param name: The name of the layer.
		:return: The layer with the given name. If the name is None, the first layer is returned.
		"""
		if name is None:
			return self.get_all_layers()[0]
		else:
			return self.get_dict_of_layers()[name]

	def __getitem__(self, name: Optional[str]) -> nn.Module:
		"""
		Get a layer of the model. If the name is None, the first layer is returned which is useful when the model has
		only one layer.
		:param name: The name of the layer.
		:return: The layer with the given name. If the name is None, the first layer is returned.
		"""
		return self.get_layer(name)

	def infer_sizes_from_inputs(self, inputs: Union[Dict[str, Any], torch.Tensor]):
		"""
		Infer the sizes of the inputs layers from the inputs of the network. The sizes of the inputs layers are set to
		the size of the inputs without the batch dimension.
		:param inputs: The inputs of the network.
		:return: None
		"""
		if isinstance(inputs, torch.Tensor):
			inputs = {
				layer_name: inputs
				for layer_name, _ in self.input_layers.items()
			}
		self.input_sizes = {k: v.shape[1:] for k, v in inputs.items()}

	def _format_spike_funcs_(
			self,
			spike_funcs: Union[Acceptable_Spike_Func, Iterable[Acceptable_Spike_Func]]
	) -> List[SpikeFunction]:
		warnings.warn(
			"The spike functions are not used anymore in the SequentialModel.",
			DeprecationWarning
		)
		if not isinstance(spike_funcs, Iterable):
			spike_funcs = [spike_funcs]
		for i, spike_func in enumerate(spike_funcs):
			if isinstance(spike_func, SpikeFuncType):
				spike_funcs[i] = SpikeFuncType2Func[spike_funcs]
		assert len(spike_funcs) == len(self.n_hidden_neurons), \
			"Number of spike functions must match number of hidden neurons"
		return spike_funcs

	def _format_layer_types_(
			self,
			layer_types: Union[Acceptable_Layer_Type, Iterable[Acceptable_Layer_Type]]
	) -> List[Type[BaseLayer]]:
		warnings.warn(
			"This function is not used anymore in the SequentialModel.",
			DeprecationWarning
		)
		if not isinstance(layer_types, Iterable):
			layer_types = [layer_types]
		for i, layer_type in enumerate(layer_types):
			layer_types[i] = self._format_layer_type_(layer_type)
		assert len(layer_types) == len(self.n_hidden_neurons), \
			"Number of layer types must match number of hidden neurons"
		return layer_types

	def _add_input_layer_(self):
		warnings.warn(
			"This function is not used anymore in the SequentialModel.",
			DeprecationWarning
		)
		if not self.n_hidden_neurons:
			return
		for l_name, in_size in self.input_sizes.items():
			self.input_layers[l_name] = self.hidden_layer_types[0](
				input_size=in_size,
				output_size=self.n_hidden_neurons[0],
				use_recurrent_connection=self.use_recurrent_connection,
				spike_func=self.spike_funcs[0],
				device=self._device,
				**self.kwargs
			)

	def _add_hidden_layers_(self):
		warnings.warn(
			"This function is not used anymore in the SequentialModel.",
			DeprecationWarning
		)
		if not self.n_hidden_neurons:
			return
		n_hidden_neurons = deepcopy(self.n_hidden_neurons)
		n_hidden_neurons[0] = n_hidden_neurons[0] * len(self.input_sizes)
		for i, hn in enumerate(n_hidden_neurons[:-1]):
			self.hidden_layers[f"hidden_{i}"] = self.hidden_layer_types[i + 1](
				input_size=hn,
				output_size=n_hidden_neurons[i + 1],
				use_recurrent_connection=self.use_recurrent_connection,
				spike_func=self.spike_funcs[i + 1],
				device=self._device,
				**self.kwargs
			)

	def _add_readout_layer(self):
		warnings.warn(
			"This function is not used anymore in the SequentialModel.",
			DeprecationWarning
		)
		if self.n_hidden_neurons:
			in_size = self.n_hidden_neurons[-1]
		else:
			in_size = np.sum([s for s in self.input_sizes.values()])
		for l_name, out_size in self.output_sizes.items():
			self.output_layers[l_name] = self.readout_layer_type(
				input_size=in_size,
				output_size=out_size,
				spike_func=self.spike_func,
				device=self._device,
				**self.kwargs
			)

	def _add_layers_(self):
		warnings.warn(
			"This function is not used anymore in the SequentialModel.",
			DeprecationWarning
		)
		self._add_input_layer_()
		self._add_hidden_layers_()
		self._add_readout_layer()

	def initialize_weights_(self):
		"""
		Initialize the weights of the layers of the model.
		:return: None
		"""
		for layer in self.get_all_layers():
			if getattr(layer, "initialize_weights_") and callable(layer.initialize_weights_):
				layer.initialize_weights_()

	def _format_single_inputs(self, inputs: torch.Tensor, time_steps: int) -> torch.Tensor:
		"""
		Check the shape of the inputs. If the shape of the inputs is (batch_size, features), a new dimension is added
		to the front of the tensor to make it (batch_size, 1, features).
		If the shape of the inputs is (batch_size, v_time_steps, features), v_time_steps must be less are equal to
		time_steps and the inputs will be padded by zeros for time steps greater than time_steps.
		:param inputs: Inputs tensor.
		:param time_steps: Number of time steps.
		:return: Formatted Input tensor.
		"""
		with torch.no_grad():
			if inputs.ndim == 2:
				inputs = torch.unsqueeze(inputs, 1)
				# inputs = inputs.repeat(1, time_steps, 1)
			assert inputs.ndim >= 3, \
				"shape of inputs must be (batch_size, time_steps, ...) or (batch_size, nb_features)"

			t_diff = time_steps - inputs.shape[1]
			assert t_diff >= 0, "inputs time steps must me less or equal to time_steps"
			if t_diff > 0:
				zero_inputs = torch.zeros(
					(inputs.shape[0], t_diff, *inputs.shape[2:]),
					dtype=torch.float32,
					device=self._device
				)
				inputs = torch.cat([inputs, zero_inputs], dim=1)
		return inputs.float()

	def _format_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
		"""
		Set the memory size of the sequential model if not already set. The default memory size is the number of
		time steps of the inputs. Return the formatted inputs formatted by self._format_single_inputs.
		:param inputs: Inputs dictionary.
		:return: Formatted inputs dictionary.
		"""
		max_time_steps = max([v.shape[1] for v in inputs.values()])
		if self._memory_size is None:
			self._memory_size = max_time_steps
		return {k: self._format_single_inputs(in_tensor, max_time_steps) for k, in_tensor in inputs.items()}

	def _inputs_to_dict(self, inputs: Union[Dict[str, Any], torch.Tensor]):
		"""
		Transform the inputs tensor into dictionary of tensors.
		:param inputs: The inputs of the network.
		:return: The transformed inputs.
		"""
		keys = list(self.input_layers.keys())
		if len(keys) == 0 and len(list(self.output_layers.keys())) > 0:
			keys = self.output_layers.keys()
		if isinstance(inputs, torch.Tensor):
			inputs = {k: inputs for k in keys}
		else:
			if set(inputs.keys()) != set(keys):
				raise ValueError("inputs must have the same keys as the input layers")
		return inputs

	def _get_time_steps_from_inputs(self, inputs: Dict[str, torch.Tensor]) -> int:
		"""
		Get the number of time steps from the inputs. Make sure that all inputs have the same number of time steps.
		:param inputs: The inputs of the network.
		:return: The number of time steps.
		"""
		time_steps_entries = [in_tensor.shape[1] for in_tensor in inputs.values()]
		assert len(set(time_steps_entries)) == 1, "inputs must have the same time steps"
		return time_steps_entries[0]

	def _pop_memory_(self, memory: List[Any]) -> List[Any]:
		"""
		Pop the memory from the list
		:param memory: List of memory
		:return: List of memory without the first element
		"""
		remove_count = len(memory) - self._memory_size
		if remove_count > 0:
			memory = memory[remove_count:]
		return memory

	def _init_hidden_states_memory(self) -> Dict[str, List]:
		"""
		Initialize the hidden states memory of the model.
		:return: The hidden states memory.
		"""
		return {
			layer_name: [None]
			for layer_name in self.get_all_layers_names()
		}

	def build_layers(self):
		"""
		Build the layers of the model.
		:return: None
		"""
		for layer in self.get_all_layers():
			if getattr(layer, "build") and callable(layer.build):
				if not getattr(layer, "is_built", False):
					layer.build()

	def build(self) -> 'SequentialModel':
		"""
		Build the network and all its layers.
		:return: None
		"""
		super(SequentialModel, self).build()
		self._infer_and_set_sizes_of_all_layers()
		self.build_layers()
		self.initialize_weights_()
		if self.foresight_time_steps > 0:
			self._map_outputs_to_inputs()
		self.device = self._device
		return self

	def _infer_and_set_sizes_of_all_layers(self):
		"""
		Infer the sizes of all layers and set them.
		:return: None
		"""
		inputs_layers_out_sum = 0
		for layer_name, layer in self.input_layers.items():
			if layer.input_size is None:
				layer.input_size = self.input_sizes[layer_name]
			if layer.output_size is None:
				layer.output_size = self._default_n_hidden_neurons
			inputs_layers_out_sum += int(layer.output_size)

		last_hidden_out_size = inputs_layers_out_sum
		for layer_idx, layer in enumerate(self.hidden_layers):
			if layer_idx == 0:
				layer.input_size = inputs_layers_out_sum
			else:
				layer.input_size = self.hidden_layers[layer_idx - 1].output_size
			if layer.output_size is None:
				layer.output_size = self._default_n_hidden_neurons
			last_hidden_out_size = int(layer.output_size)

		for layer_name, layer in self.output_layers.items():
			if layer.input_size is None:
				layer.input_size = last_hidden_out_size
			if layer.output_size is None:
				if self.output_sizes is None or self.output_sizes[layer_name] is None:
					warnings.warn(
						f"output_size is not set for layer {layer_name}. It will be set to {last_hidden_out_size}"
					)
					layer.output_size = last_hidden_out_size
				else:
					layer.output_size = self.output_sizes[layer_name]
			if self.output_sizes is None:
				self.output_sizes = {layer_name: layer.output_size}
			else:
				self.output_sizes[layer_name] = layer.output_size

	def _map_outputs_to_inputs(self) -> Dict[str, str]:
		"""
		Map the outputs of the model to the inputs of the model for forcasting purposes.
		:return:
		"""
		self._outputs_to_inputs_names_map = {}
		if len(self.input_layers) == 1 and len(self.output_layers) == 1:
			in_name = list(self.input_layers.keys())[0]
			out_name = list(self.output_layers.keys())[0]
			self._outputs_to_inputs_names_map[out_name] = in_name
			assert self.input_sizes[in_name] == self.output_sizes[out_name], \
				f"input ({self.input_sizes[in_name]}) and output ({self.output_sizes[out_name]}) sizes must be the " \
				f"same when foresight_time_steps > 0."
		elif len(self.input_layers) == 0 and len(self.output_layers) >= 1:
			for out_layer_name in self._ordered_outputs_names:
				self._outputs_to_inputs_names_map[out_layer_name] = out_layer_name
		else:
			self._outputs_to_inputs_names_map: Dict[str, str] = {
				out_layer_name: in_layer_name
				for in_layer_name, out_layer_name in zip(self._ordered_inputs_names, self._ordered_outputs_names)
			}
			for out_layer_name, in_layer_name in self._outputs_to_inputs_names_map.items():
				assert self.input_sizes[in_layer_name] == self.output_sizes[out_layer_name], \
					"input and output sizes must be the same when foresight_time_steps > 0."
		return self._outputs_to_inputs_names_map

	def _inputs_forward_(
			self,
			inputs: Dict[str, torch.Tensor],
			hidden_states: Dict[str, List],
			t: int
	) -> torch.Tensor:
		features_list = []
		for layer_name, layer in self.input_layers.items():
			hh = hidden_states[layer.name][-1] if hidden_states[layer.name] else None
			features, hh = layer(inputs[layer_name][:, t], hh)
			hidden_states[layer_name].append(hh)
			features_list.append(features)
		if features_list:
			forward_tensor = torch.concat(features_list, dim=1)
		else:
			forward_tensor = torch.concat([inputs[in_name][:, t] for in_name in inputs], dim=1)
		return forward_tensor

	def _hidden_forward_(
			self,
			forward_tensor: torch.Tensor,
			hidden_states: Dict[str, List],
	) -> torch.Tensor:
		for layer_idx, layer in enumerate(self.hidden_layers):
			hh = hidden_states[layer.name][-1] if hidden_states[layer.name] else None
			forward_tensor, hh = layer(forward_tensor, hh)
			hidden_states[layer.name].append(hh)
		return forward_tensor

	def _readout_forward_(
			self,
			forward_tensor: torch.Tensor,
			hidden_states: Dict[str, List],
			outputs_trace: Dict[str, List[torch.Tensor]]
	):
		for layer_name, layer in self.output_layers.items():
			hh = hidden_states[layer_name][-1] if hidden_states[layer_name] else None
			out, hh = layer(forward_tensor, hh)
			outputs_trace[layer_name].append(out)
			hidden_states[layer_name].append(hh)
		return outputs_trace

	def forward(
			self,
			inputs: Union[Dict[str, Any], torch.Tensor],
			**kwargs
	) -> Tuple[Dict[str, torch.Tensor], Dict[str, Tuple[torch.Tensor, ...]]]:
		"""
		Forward pass of the model.
		-> When it comes to integrate a time series:
			* We integrate the initial conditions <time_step> times.
			* We predict the remaining <forward_sight_time_steps - 1> time steps from the initial conditions
			* Please note that the last output of the integration of the initial conditions is the input for
			the integration of the remaining time steps AND also the first prediction.
			Example: time_series = [t_0, t_1 ... t_N] if:
				[t_0, t_1] -> Initial conditions, then t_1 generate the first prediction (t_2) :
				[t_2, t_3 ... t_N] -> The remaining time steps are predicted from the initial conditions.

		:param inputs: The inputs to the model where the dimensions are
						{input_name: (batch_size, time_steps, input_size)}. If the inputs have the shape
						(batch_size, input_size), then the time_steps is 1. All the inputs must have the same
						time_steps otherwise the inputs with lower time_steps will be padded with zeros.
		:param kwargs: Additional arguments for the forward pass.
		:return: A tuple of two dictionaries. The first dictionary contains the outputs of the model and the second
						dictionary contains the hidden states of the model. The keys of the dictionaries are the
						names of the layers. The values of the dictionaries are lists of tensors. The length of the
						lists is the number of time steps.
		"""
		inputs = self._inputs_to_dict(inputs)
		inputs = self.apply_input_transform(inputs)
		inputs = self._format_inputs(inputs)
		time_steps = self._get_time_steps_from_inputs(inputs)
		hidden_states = self._init_hidden_states_memory()
		outputs_trace: Dict[str, List[torch.Tensor]] = defaultdict(list)

		# TODO: implement a way to integrate an other time dimension for inputs of
		#  shape: (batch_size, time_steps, time_steps, ...). Those inputs are obtained for forecasting a time series of
		#  real values that were transformed to times series of spikes.

		# integration of the inputs or the initial conditions
		for t in range(time_steps):
			forward_tensor = self._inputs_forward_(inputs, hidden_states, t)
			forward_tensor = self._hidden_forward_(forward_tensor, hidden_states)
			outputs_trace = self._readout_forward_(forward_tensor, hidden_states, outputs_trace)

			outputs_trace = {layer_name: self._pop_memory_(trace) for layer_name, trace in outputs_trace.items()}
			hidden_states = {layer_name: self._pop_memory_(trace) for layer_name, trace in hidden_states.items()}

		# Foresight prediction of the initial conditions
		for t in range(self.foresight_time_steps-1):
			foresight_inputs_tensor = {
				self._outputs_to_inputs_names_map[layer_name]: torch.stack(trace, dim=1)
				for layer_name, trace in outputs_trace.items()
			}
			forward_tensor = self._inputs_forward_(foresight_inputs_tensor, hidden_states, -1)
			forward_tensor = self._hidden_forward_(forward_tensor, hidden_states)
			outputs_trace = self._readout_forward_(forward_tensor, hidden_states, outputs_trace)

		hidden_states = self._format_hidden_outputs_traces(hidden_states)
		outputs_trace_tensor = {layer_name: torch.stack(trace, dim=1) for layer_name, trace in outputs_trace.items()}
		outputs_trace_tensor = self.apply_output_transform(outputs_trace_tensor)
		return outputs_trace_tensor, hidden_states

	def get_prediction_trace(
			self,
			inputs: Union[Dict[str, Any], torch.Tensor],
			**kwargs
	) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
		"""
		Returns the prediction trace for the given inputs. Method used for time series prediction.
		
		:param inputs: inputs to the network.
		:param kwargs: kwargs to be passed to the forward method.
		
		:keyword int foresight_time_steps: number of time steps to predict. Default is self.foresight_time_steps.
		
		:return: the prediction trace.
		"""
		foresight_time_steps = kwargs.get('foresight_time_steps', self.foresight_time_steps)
		outputs_trace, hidden_states = self(inputs.to(self.device), **kwargs)
		if isinstance(outputs_trace, dict):
			outputs_trace = {
				layer_name: trace[:, -foresight_time_steps:]
				for layer_name, trace in outputs_trace.items()
			}
			if len(outputs_trace) == 1:
				return outputs_trace[list(outputs_trace.keys())[0]]
		else:
			outputs_trace = outputs_trace[:, -foresight_time_steps:]
		return outputs_trace

	def get_raw_prediction(
			self,
			inputs: torch.Tensor,
			re_outputs_trace: bool = True,
			re_hidden_states: bool = True
	) -> Union[Tuple[Any, Any, Any], Tuple[Any, Any], Any]:
		outputs_trace, hidden_states = self(inputs.to(self._device))
		if isinstance(outputs_trace, torch.Tensor):
			logits, _ = torch.max(outputs_trace, dim=1)
		elif isinstance(outputs_trace, dict):
			logits = {
				k: torch.max(v, dim=1)[0]
				for k, v in outputs_trace.items()
			}
		else:
			raise ValueError("outputs_trace must be a torch.Tensor or a dictionary")
		# logits = batchwise_temporal_filter(outputs_trace, decay=0.9)
		if re_outputs_trace and re_hidden_states:
			return logits, outputs_trace, hidden_states
		elif re_outputs_trace:
			return logits, outputs_trace
		elif re_hidden_states:
			return logits, hidden_states
		else:
			return logits

	def get_prediction_proba(
			self,
			inputs: torch.Tensor,
			re_outputs_trace: bool = True,
			re_hidden_states: bool = True
	) -> Union[Tuple[Any, Any, Any], Tuple[Any, Any], Any]:
		outs = self.get_raw_prediction(inputs, re_outputs_trace, re_hidden_states)
		if isinstance(outs, (list, tuple)):
			m = outs[0]
		else:
			m = outs
		if isinstance(m, torch.Tensor):
			proba = torch.softmax(m, dim=-1)
		elif isinstance(m, dict):
			proba = {
				k: torch.softmax(v, dim=-1)
				for k, v in m.items()
			}
		else:
			raise ValueError("m must be a torch.Tensor or a dictionary")
		if re_outputs_trace or re_hidden_states:
			return proba, *outs[1:]
		return proba

	def get_prediction_log_proba(
			self,
			inputs: torch.Tensor,
			re_outputs_trace: bool = True,
			re_hidden_states: bool = True
	) -> Union[tuple[Tensor, Any, Any], tuple[Tensor, Any], Tensor]:
		outs = self.get_raw_prediction(inputs, re_outputs_trace, re_hidden_states)
		if isinstance(outs, (list, tuple)):
			m = outs[0]
		else:
			m = outs
		if isinstance(m, torch.Tensor):
			log_proba = F.log_softmax(m, dim=-1)
		elif isinstance(m, dict):
			log_proba = {
				k: F.log_softmax(v, dim=-1)
				for k, v in m.items()
			}
		else:
			raise ValueError("m must be a torch.Tensor or a dictionary")
		if re_outputs_trace or re_hidden_states:
			return log_proba, *outs[1:]
		return log_proba

	def get_and_reset_regularization_loss(self) -> torch.Tensor:
		"""
		Get the regularization loss as a sum of all the regularization losses of the layers. Then reset the
		regularization losses.
		:return: the regularization loss.
		"""
		regularization_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
		for layer in self.get_all_layers():
			if hasattr(layer, "get_and_reset_regularization_loss") and callable(layer.get_and_reset_regularization_loss):
				regularization_loss += layer.get_and_reset_regularization_loss()
		return regularization_loss


