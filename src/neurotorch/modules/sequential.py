import json
import logging
import os
import shutil
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from tqdm.auto import tqdm
from torch import Tensor, nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from . import (
	BaseModel,
	HeavisideSigmoidApprox,
	BaseLayer,
	LIFLayer,
	LayerType,
	LayerType2Layer,
	LILayer,
	SpikeFuncType,
	SpikeFuncType2Func,
	SpikeFunction
)
from ..callbacks import LoadCheckpointMode
from ..dimension import Dimension

Acceptable_Spike_Func = Union[Type[SpikeFunction], SpikeFuncType]
Acceptable_Spike_Funcs = Union[Acceptable_Spike_Func, Iterable[Acceptable_Spike_Func]]
Acceptable_Layer_Type = Union[Type[BaseLayer], LayerType]
Acceptable_Layer_Types = Union[Acceptable_Layer_Type, Iterable[Acceptable_Layer_Type]]
IntDimension = Union[int, Dimension]


class SequentialModel(BaseModel):
	def __new__(
			cls,
			*,
			input_sizes: Optional[Union[Dict[str, IntDimension], List[int], IntDimension]] = None,
			output_sizes: Optional[Union[Dict[str, int], List[int], int]] = None,
			n_hidden_neurons: Optional[Union[int, Iterable[int]]] = None,
			use_recurrent_connection: Optional[Union[bool, Iterable[bool]]] = None,
			spike_funcs: Optional[Acceptable_Spike_Funcs] = None,
			hidden_layer_types: Optional[Acceptable_Layer_Types] = None,
			readout_layer_type: Optional[Acceptable_Layer_Type] = None,
			layers: Optional[Iterable[BaseLayer]] = None,
			**kwargs
	):
		# TODO: if the first argument is a iterable of layer, juste call the constructor
		# TODO: instead create the iterable of layers with the named parameters and call the constructor
		# TODO: the call of the constructor should be like this:
		# model = Sequential(
		#   input_layers=...
		#   hidden_layers=...
		#   output_layers=...
		# )
		# exemple: https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-implement-multiple-constructors
		auto_construct_parameters = [
			input_sizes,
			output_sizes,
			n_hidden_neurons,
			use_recurrent_connection,
			spike_funcs,
			hidden_layer_types,
			readout_layer_type
		]
		if layers is not None:
			assert all([p is None for p in auto_construct_parameters]), \
				f"You can't use the named parameters: {auto_construct_parameters} and layers at the same time"
			return super(SequentialModel, cls).__new__(cls)
		raise NotImplementedError("Auto construct feature is not available yet.")
	
	def __init__(
			self,
			layers: Iterable[Union[Iterable[BaseLayer], BaseLayer]],
			int_time_steps: int = 100,
			name: str = "snn",
			checkpoint_folder: str = "checkpoints",
			device: Optional[torch.device] = None,
			input_transform: Optional[Union[Dict[str, Callable], List[Callable]]] = None,
			**kwargs
	):
		input_layers, hidden_layers, output_layers = self._format_layers(layers)
		super(SequentialModel, self).__init__(
			input_sizes={layer.name: layer._input_size for _, layer in input_layers.items()},
			output_size={layer.name: layer._output_size for _, layer in output_layers.items()},
			name=name,
			checkpoint_folder=checkpoint_folder,
			device=device,
			input_transform=input_transform,
			**kwargs
		)
		self.input_layers, self.hidden_layers, self.output_layers = self._layers_containers_to_modules(
			input_layers, hidden_layers, output_layers
		)
		assert len(self.get_all_layers_names()) == len(set(self.get_all_layers_names())), \
			"There are layers with the same name."
		self.int_time_steps = int_time_steps
		# self.n_hidden_neurons = self._format_hidden_neurons_(n_hidden_neurons)
		# self.spike_func = self._format_spike_funcs_(spike_funcs)
		# self.hidden_layer_types: List[Type] = self._format_layer_types_(hidden_layer_types)
		# self.readout_layer_type = self._format_layer_type_(readout_layer_type)  # TODO: change for multiple readout layers
		# self._add_layers_()
		self.initialize_weights_()
		self._memory_size = self.kwargs.get("memory_size", self.int_time_steps)
		assert self._memory_size > 0, "The memory size must be greater than 0."
	
	def get_all_layers(self) -> List[nn.Module]:
		return list(self.input_layers.values()) + list(self.hidden_layers) + list(self.output_layers.values())
	
	def get_all_layers_names(self) -> List[str]:
		return [layer.name for layer in self.get_all_layers()]
	
	@staticmethod
	def _format_input_output_layers(layers: Iterable[Union[Iterable[BaseLayer], BaseLayer]]) -> Dict:
		layers: Iterable[BaseLayer] = [layers] if not isinstance(layers, Iterable) else layers
		assert all([isinstance(layer, BaseLayer) for layer in layers]), \
			"All layers must be of type BaseLayer"
		if not isinstance(layers, dict):
			layers = {layer.name: layer for layer in layers}
		return layers
	
	@staticmethod
	def _format_layers(
			layers: Iterable[Union[Iterable[BaseLayer], BaseLayer]]
	) -> Tuple[Dict, List, Dict]:
		if not isinstance(layers, Iterable):
			layers = [layers]
		if len(layers) > 1:
			input_layers = SequentialModel._format_input_output_layers(layers[0])
		else:
			input_layers = nn.ModuleDict()
		
		if len(layers) > 2:
			hidden_layers = layers[1:-1]
			if not isinstance(hidden_layers, Iterable):
				hidden_layers = [hidden_layers]
			assert all([isinstance(layer, BaseLayer) for layer in hidden_layers]), \
				"All hidden layers must be of type BaseLayer"
		else:
			hidden_layers = []
		
		output_layers = SequentialModel._format_input_output_layers(layers[-1])
		return input_layers, hidden_layers, output_layers
	
	@staticmethod
	def _layers_containers_to_modules(
			inputs_layers: Dict,
			hidden_layers: List,
			outputs_layers: Dict
	) -> Tuple[nn.ModuleDict, nn.ModuleList, nn.ModuleDict]:
		input_layers = nn.ModuleDict(inputs_layers)
		hidden_layers = nn.ModuleList(hidden_layers)
		output_layers = nn.ModuleDict(outputs_layers)
		return input_layers, hidden_layers, output_layers
	
	def _format_spike_funcs_(
			self,
			spike_funcs: Union[Acceptable_Spike_Func, Iterable[Acceptable_Spike_Func]]
	) -> List[SpikeFunction]:
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
		if not isinstance(layer_types, Iterable):
			layer_types = [layer_types]
		for i, layer_type in enumerate(layer_types):
			layer_types[i] = self._format_layer_type_(layer_type)
		assert len(layer_types) == len(self.n_hidden_neurons), \
			"Number of layer types must match number of hidden neurons"
		return layer_types
	
	@staticmethod
	def _format_layer_type_(
			layer_type: Optional[Acceptable_Layer_Type]
	) -> Optional[Type[BaseLayer]]:
		if isinstance(layer_type, LayerType):
			layer_type = LayerType2Layer[layer_type]
		return layer_type
	
	@staticmethod
	def _format_hidden_neurons_(n_hidden_neurons: Optional[Union[int, Iterable[int]]]) -> List[int]:
		if n_hidden_neurons is None:
			return []
		if not isinstance(n_hidden_neurons, Iterable):
			n_hidden_neurons = [n_hidden_neurons]
		return n_hidden_neurons
	
	def _add_input_layer_(self):
		if not self.n_hidden_neurons:
			return
		for l_name, in_size in self.input_sizes.items():
			self.input_layers[l_name] = self.hidden_layer_types[0](
				input_size=in_size,
				output_size=self.n_hidden_neurons[0],
				use_recurrent_connection=self.use_recurrent_connection,
				spike_func=self.spike_funcs[0],
				device=self.device,
				**self.kwargs
			)
	
	def _add_hidden_layers_(self):
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
				device=self.device,
				**self.kwargs
			)
	
	def _add_readout_layer(self):
		if self.n_hidden_neurons:
			in_size = self.n_hidden_neurons[-1]
		else:
			in_size = np.sum([s for s in self.input_sizes.values()])
		for l_name, out_size in self.output_sizes.items():
			self.output_layers[l_name] = self.readout_layer_type(
				input_size=in_size,
				output_size=out_size,
				spike_func=self.spike_func,
				device=self.device,
				**self.kwargs
			)
	
	def _add_layers_(self):
		self._add_input_layer_()
		self._add_hidden_layers_()
		self._add_readout_layer()
	
	def initialize_weights_(self):
		for param in self.parameters():
			if param.ndim > 2:
				torch.nn.init.xavier_normal_(param)
			else:
				torch.nn.init.normal_(param)
		for layer in self.get_all_layers():
			if getattr(layer, "initialize_weights_") and callable(layer.initialize_weights_):
				layer.initialize_weights_()
	
	def _format_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
		"""
		Check the shape of the inputs. If the shape of the inputs is (batch_size, features),
		the inputs is considered constant over time and the inputs will be repeat over self.int_time_steps time steps.
		If the shape of the inputs is (batch_size, time_steps, features), time_steps must be less are equal to
		self.int_time_steps and the inputs will be padded by zeros for time steps greater than time_steps.
		:param inputs: Inputs tensor
		:return: Formatted Input tensor.
		"""
		# TODO: adapt to DimensionProperty
		with torch.no_grad():
			if inputs.ndim == 2:
				inputs = torch.unsqueeze(inputs, 1)
				inputs = inputs.repeat(1, self.int_time_steps, 1)
			assert inputs.ndim == 3, \
				"shape of inputs must be (batch_size, time_steps, nb_features) or (batch_size, nb_features)"
			
			t_diff = self.int_time_steps - inputs.shape[1]
			assert t_diff >= 0, "inputs time steps must me less or equal to int_time_steps"
			if t_diff > 0:
				zero_inputs = torch.zeros(
					(inputs.shape[0], t_diff, inputs.shape[-1]),
					dtype=torch.float32,
					device=self.device
				)
				inputs = torch.cat([inputs, zero_inputs], dim=1)
		return inputs.float()
	
	def _format_hidden_outputs_traces(
			self,
			hidden_states: Dict[str, List[Tuple[torch.Tensor, ...]]]
	) -> Dict[str, Tuple[torch.Tensor, ...]]:
		"""
		Permute the hidden states to have a dictionary of shape {layer_name: (tensor, ...)}
		:param hidden_states: Dictionary of hidden states
		:return: Dictionary of hidden states with the shape {layer_name: (tensor, ...)}
		"""
		hidden_states = {
			layer_name: tuple([torch.stack(e, dim=1) for e in list(zip(*trace))])
			for layer_name, trace in hidden_states.items()
		}
		return hidden_states
	
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
		return {
			# layer_name: [None for t in range(self.int_time_steps+1)]
			layer_name: [None]
			for layer_name in self.get_all_layers_names()
		}
	
	def _inputs_forward_(
			self,
			inputs: Dict[str, torch.Tensor],
			hidden_states: Dict[str, List],
			t: int
	) -> torch.Tensor:
		features_list = []
		for layer_name, layer in self.input_layers.items():
			features, hh = layer(inputs[layer_name][:, t])
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
		if isinstance(inputs, torch.Tensor):
			inputs = {k: inputs for k in self.input_layers.keys()}
		inputs = self.apply_transform(inputs)
		inputs = {k: self._format_inputs(in_tensor) for k, in_tensor in inputs.items()}
		hidden_states = self._init_hidden_states_memory()
		outputs_trace: Dict[str, List[torch.Tensor]] = defaultdict(list)
		
		for t in range(self.int_time_steps):
			forward_tensor = self._inputs_forward_(inputs, hidden_states, t)
			forward_tensor = self._hidden_forward_(forward_tensor, hidden_states)
			outputs_trace = self._readout_forward_(forward_tensor, hidden_states, outputs_trace)
			
			outputs_trace = {layer_name: self._pop_memory_(trace) for layer_name, trace in outputs_trace.items()}
			hidden_states = {layer_name: self._pop_memory_(trace) for layer_name, trace in hidden_states.items()}
		
		# hidden_states = {layer_name: trace[1:] for layer_name, trace in hidden_states.items()}
		hidden_states = self._format_hidden_outputs_traces(hidden_states)
		outputs_trace_tensor = {layer_name: torch.stack(trace, dim=1) for layer_name, trace in outputs_trace.items()}
		return outputs_trace_tensor, hidden_states
	
	def get_raw_prediction(
			self,
			inputs: torch.Tensor,
			re_outputs_trace: bool = True,
			re_hidden_states: bool = True
	) -> Union[Tuple[Any, Any, Any], Tuple[Any, Any], Any]:
		outputs_trace, hidden_states = self(inputs.to(self.device))
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
			return proba, outs[1:]
		return proba
	
	def get_prediction_log_proba(
			self,
			inputs: torch.Tensor,
			re_outputs_trace: bool = True,
			re_hidden_states: bool = True
	) -> Union[tuple[Tensor, Any, Any], tuple[Tensor, Any], Tensor]:
		m, *outs = self.get_raw_prediction(inputs, re_outputs_trace, re_hidden_states)
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
			return log_proba, *outs
		return log_proba
	
	def get_spikes_count_per_neuron(self, hidden_states: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
		"""
		Get the spikes count per neuron from the hidden states
		:return:
		"""
		counts = []
		for l_name, traces in hidden_states.items():
			if isinstance(self.hidden_layers[l_name], LIFLayer):
				counts.extend(traces[-1].sum(dim=(0, 1)).tolist())
		return torch.tensor(counts, dtype=torch.float32, device=self.device)

# def plot_loss_history(self, loss_history: LossHistory = None, show=False):
# 	if loss_history is None:
# 		loss_history = self.loss_history
# 	save_path = f"./{self.checkpoint_folder}/loss_history.png"
# 	os.makedirs(f"./{self.checkpoint_folder}/", exist_ok=True)
# 	loss_history.plot(save_path, show)

# def _create_checkpoint_path(self, epoch: int = -1):
# 	return f"./{self.checkpoint_folder}/{self.model_name}{SequentialModel.SUFFIX_SEP}{SequentialModel.CHECKPOINT_EPOCH_KEY}{epoch}{SequentialModel.SAVE_EXT}"
#
# def _create_new_checkpoint_meta(self, epoch: int, best: bool = False) -> dict:
# 	save_path = self._create_checkpoint_path(epoch)
# 	new_info = {SequentialModel.CHECKPOINT_EPOCHS_KEY: {epoch: save_path}}
# 	if best:
# 		new_info[SequentialModel.CHECKPOINT_BEST_KEY] = save_path
# 	return new_info

# def save_checkpoint(
# 		self,
# 		optimizer,
# 		epoch: int,
# 		epoch_losses: Dict[str, Any],
# 		best: bool = False,
# ):
# 	os.makedirs(self.checkpoint_folder, exist_ok=True)
# 	save_path = self._create_checkpoint_path(epoch)
# 	torch.save({
# 		SequentialModel.CHECKPOINT_EPOCH_KEY: epoch,
# 		SequentialModel.CHECKPOINT_STATE_DICT_KEY: self.state_dict(),
# 		SequentialModel.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: optimizer.state_dict(),
# 		SequentialModel.CHECKPOINT_LOSS_KEY: epoch_losses,
# 	}, save_path)
# 	self.save_checkpoints_meta(self._create_new_checkpoint_meta(epoch, best))

# @staticmethod
# def get_save_path_from_checkpoints(
# 		checkpoints_meta: Dict[str, Union[str, Dict[Any, str]]],
# 		load_checkpoint_mode: LoadCheckpointMode = LoadCheckpointMode.BEST_EPOCH
# ) -> str:
# 	if load_checkpoint_mode == load_checkpoint_mode.BEST_EPOCH:
# 		return checkpoints_meta[SequentialModel.CHECKPOINT_BEST_KEY]
# 	elif load_checkpoint_mode == load_checkpoint_mode.LAST_EPOCH:
# 		epochs_dict = checkpoints_meta[SequentialModel.CHECKPOINT_EPOCHS_KEY]
# 		last_epoch: int = max([int(e) for e in epochs_dict])
# 		return checkpoints_meta[SequentialModel.CHECKPOINT_EPOCHS_KEY][str(last_epoch)]
# 	else:
# 		raise ValueError()
#
# def get_checkpoints_loss_history(self) -> LossHistory:
# 	history = LossHistory()
# 	with open(self.checkpoints_meta_path, "r+") as jsonFile:
# 		meta: dict = json.load(jsonFile)
# 	checkpoints = [torch.load(path) for path in meta[SequentialModel.CHECKPOINT_EPOCHS_KEY].values()]
# 	for checkpoint in checkpoints:
# 		history.concat(checkpoint[SequentialModel.CHECKPOINT_LOSS_KEY])
# 	return history
#
# def load_checkpoint(
# 		self,
# 		load_checkpoint_mode: LoadCheckpointMode = LoadCheckpointMode.BEST_EPOCH
# ) -> dict:
# 	with open(self.checkpoints_meta_path, "r+") as jsonFile:
# 		info: dict = json.load(jsonFile)
# 	path = self.get_save_path_from_checkpoints(info, load_checkpoint_mode)
# 	checkpoint = torch.load(path)
# 	self.load_state_dict(checkpoint[SequentialModel.CHECKPOINT_STATE_DICT_KEY], strict=True)
# 	return checkpoint

# def save_checkpoints_meta(self, new_info: dict):
# 	info = dict()
# 	if os.path.exists(self.checkpoints_meta_path):
# 		with open(self.checkpoints_meta_path, "r+") as jsonFile:
# 			info = json.load(jsonFile)
# 	mapping_update_recursively(info, new_info)
# 	with open(self.checkpoints_meta_path, "w+") as jsonFile:
# 		json.dump(info, jsonFile, indent=4)

# def compute_confusion_matrix(
# 		self,
# 		nb_classes: int,
# 		dataloaders: Dict[str, DataLoader],
# 		fit=False,
# 		fit_kwargs=None,
# 		load_checkpoint_mode: LoadCheckpointMode = None,
# ):
# 	if fit_kwargs is None:
# 		fit_kwargs = {}
# 	if fit:
# 		self.fit(dataloaders['train'], dataloaders['val'], **fit_kwargs)
#
# 	if load_checkpoint_mode is not None:
# 		self.load_checkpoint(load_checkpoint_mode)
# 	return {key: self._compute_single_confusion_matrix(nb_classes, d) for key, d in dataloaders.items()}

# def _compute_single_confusion_matrix(self, nb_classes: int, dataloader: DataLoader) -> np.ndarray:
# 	self.eval()
# 	confusion_matrix = np.zeros((nb_classes, nb_classes))
# 	with torch.no_grad():
# 		for i, (inputs, classes) in enumerate(dataloader):
# 			inputs = inputs.to(self.device)
# 			classes = classes.to(self.device)
# 			outputs = self.get_prediction_logits(inputs, re_outputs_trace=False, re_hidden_states=False)
# 			_, preds = torch.max(outputs, -1)
# 			for t, p in zip(classes.view(-1), preds.view(-1)):
# 				confusion_matrix[t.long(), p.long()] += 1
# 	return confusion_matrix
