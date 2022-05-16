import json
import logging
import os
import shutil
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
	RNNLayer,
	LIFLayer,
	LayerType,
	LayerType2Layer,
	LILayer,
	SpikeFuncType,
	SpikeFuncType2Func,
	SpikeFunction
)

Acceptable_Spike_Func = Union[Type[SpikeFunction], SpikeFuncType]
Acceptable_Layer_Type = Union[Type[RNNLayer], LayerType]


class SequentialModel(BaseModel):
	def __new__(cls, *args, **kwargs):
		# TODO: if the first argument is a iterable of layer, juste call the constructor
		# TODO: instead create the iterable of layers with the named parameters and call the constructor
		# TODO: the call of the constructor should be like this:
		# model = Sequential(
		#   input_layers=...
		#   hidden_layers=...
		#   output_layers=...
		# )
		# exemple: https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-implement-multiple-constructors
		raise NotImplementedError
	
	def __init__(
			self,
			# input_sizes: Union[Dict[str, int], List[int], int],
			# output_sizes: Union[Dict[str, int], List[int], int],
			# n_hidden_neurons: Optional[Union[int, Iterable[int]]] = None,
			# use_recurrent_connection: Union[bool, Iterable[bool]] = True,
			# spike_funcs: Union[Acceptable_Spike_Func, Iterable[Acceptable_Spike_Func]] = HeavisideSigmoidApprox,
			# hidden_layer_types: Union[Acceptable_Layer_Type, Iterable[Acceptable_Layer_Type]] = LIFLayer,
			# readout_layer_type: Optional[Acceptable_Layer_Type] = LILayer,
			layers: Iterable[RNNLayer],  # TODO: change the inputs to be more flexible
			int_time_steps: int = 100,
			name: str = "snn",
			checkpoint_folder: str = "checkpoints",
			device: Optional[torch.device] = None,
			input_transform: Optional[Union[Dict[str, Callable], List[Callable]]] = None,
			**kwargs
	):
		super(SequentialModel, self).__init__(
			input_sizes=input_sizes,
			output_size=output_sizes,
			name=name,
			checkpoint_folder=checkpoint_folder,
			device=device,
			input_transform=input_transform,
			**kwargs
		)
		self.dt = self.kwargs.pop("dt", 1e-3)
		self.int_time_steps = int_time_steps
		self.n_hidden_neurons = self._format_hidden_neurons_(n_hidden_neurons)
		self.spike_func = self._format_spike_funcs_(spike_funcs)
		self.hidden_layer_types: List[Type] = self._format_layer_types_(hidden_layer_types)
		self.readout_layer_type = self._format_layer_type_(readout_layer_type)  # TODO: change for multiple readout layers
		self.use_recurrent_connection = use_recurrent_connection
		self.input_layers = nn.ModuleDict()
		self.hidden_layers = nn.ModuleDict()
		self.output_layers = nn.ModuleDict()
		self._add_layers_()
		self.initialize_weights_()

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
	) -> List[Type[RNNLayer]]:
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
	) -> Optional[Type[RNNLayer]]:
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
				dt=self.dt,
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
		for layer_name, layer in self.hidden_layers.items():
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

	def _format_hidden_outputs(
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

	def forward(self, inputs: Dict[str, Any], **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
		inputs = self._format_inputs(inputs)
		hidden_states = {
			layer_name: [None for t in range(self.int_time_steps+1)]
			for layer_name, _ in self.hidden_layers.items()
		}
		outputs_trace: List[torch.Tensor] = []

		for t in range(1, self.int_time_steps+1):
			forward_tensor = inputs[:, t-1]
			for layer_idx, (layer_name, layer) in enumerate(self.hidden_layers.items()):
				hh = hidden_states[layer_name][t - 1]
				forward_tensor, hidden_states[layer_name][t] = layer(forward_tensor, hh)
			outputs_trace.append(forward_tensor)

		hidden_states = {layer_name: trace[1:] for layer_name, trace in hidden_states.items()}
		hidden_states = self._format_hidden_outputs(hidden_states)
		outputs_trace_tensor = torch.stack(outputs_trace, dim=1)
		return outputs_trace_tensor, hidden_states

	def get_prediction_logits(
			self,
			inputs: torch.Tensor,
			re_outputs_trace: bool = True,
			re_hidden_states: bool = True
	) -> Union[tuple[Tensor, Any, Any], tuple[Tensor, Any], Tensor]:
		outputs_trace, hidden_states = self(inputs.to(self.device))
		logits, _ = torch.max(outputs_trace, dim=1)
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
	) -> Union[tuple[Tensor, Any, Any], tuple[Tensor, Any], Tensor]:
		m, *outs = self.get_prediction_logits(inputs, re_outputs_trace, re_hidden_states)
		if re_outputs_trace or re_hidden_states:
			return F.softmax(m, dim=-1), *outs
		return m

	def get_prediction_log_proba(
			self,
			inputs: torch.Tensor,
			re_outputs_trace: bool = True,
			re_hidden_states: bool = True
	) -> Union[tuple[Tensor, Any, Any], tuple[Tensor, Any], Tensor]:
		m, *outs = self.get_prediction_logits(inputs, re_outputs_trace, re_hidden_states)
		if re_outputs_trace or re_hidden_states:
			return F.log_softmax(m, dim=-1), *outs
		return m

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

	def _check_early_stopping(self, patience: int, tol: float = 1e-2) -> bool:
		"""
		:param patience:
		:return:
		"""
		losses = self.loss_history['val'][-patience:]
		return np.all(np.abs(np.diff(losses)) < tol)

	def fit(
			self,
			train_dataloader: DataLoader,
			val_dataloader: DataLoader,
			lr=1e-3,
			nb_epochs=15,
			criterion=None,
			optimizer=None,
			load_checkpoint_mode: LoadCheckpointMode = None,
			force_overwrite: bool = False,
			early_stopping: bool = False,
			early_stopping_patience: int = 5,
			verbose: bool = True,
			p_bar_position: Optional[int] = None,
			p_bar_leave: Optional[bool] = None,
	):
		if criterion is None:
			criterion = nn.NLLLoss()
		if optimizer is None:
			optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)

		start_epoch = 0
		if load_checkpoint_mode is None:
			assert os.path.exists(self.checkpoints_meta_path) or force_overwrite, \
				f"{self.checkpoints_meta_path} already exists. " \
				f"Set force_overwrite flag to True to overwrite existing saves."
			if os.path.exists(self.checkpoints_meta_path) and force_overwrite:
				shutil.rmtree(self.checkpoint_folder)
		else:
			try:
				checkpoint = self.load_checkpoint(load_checkpoint_mode)
				self.load_state_dict(checkpoint[SequentialModel.CHECKPOINT_STATE_DICT_KEY], strict=True)
				optimizer.load_state_dict(checkpoint[SequentialModel.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY])
				start_epoch = int(checkpoint[SequentialModel.CHECKPOINT_EPOCH_KEY]) + 1
				self.loss_history = self.get_checkpoints_loss_history()
			except FileNotFoundError:
				if verbose:
					logging.warning("No such checkpoint. Fit from beginning.", UserWarning)

		if start_epoch >= nb_epochs:
			return self.loss_history

		best_loss = self.loss_history.min('val')
		p_bar = tqdm(
			range(start_epoch, nb_epochs),
			desc="Training",
			disable=not verbose,
			position=p_bar_position,
			unit="epoch",
			leave=p_bar_leave
		)
		for epoch in p_bar:
			epoch_loss = self._exec_phase(train_dataloader, val_dataloader, criterion, optimizer)
			epoch_val_acc = self.compute_classification_accuracy(val_dataloader, verbose=False)
			self.loss_history.concat(epoch_loss)
			is_best = epoch_loss['val'] < best_loss
			self.save_checkpoint(optimizer, epoch, epoch_loss, is_best)
			if is_best:
				best_loss = epoch_loss['val']
			p_bar.set_postfix(
				train_loss=f"{epoch_loss['train']:.5e}",
				val_loss=f"{epoch_loss['val']:.5e}",
				val_acc=f"{epoch_val_acc:.5f}",
			)
			if early_stopping and self._check_early_stopping(early_stopping_patience):
				if verbose:
					logging.info(f"Early stopping stopped the training at epoch {epoch}.")
				break
		p_bar.close()
		self.plot_loss_history(show=False)
		return self.loss_history

	def _exec_phase(self, train_dataloader, val_dataloader, criterion, optimizer):
		self.train()
		train_loss = self._exec_epoch(
			train_dataloader,
			criterion,
			optimizer,
		)
		self.eval()
		val_loss = self._exec_epoch(
			val_dataloader,
			criterion,
			optimizer,
		)
		return dict(train=train_loss, val=val_loss)

	def _exec_epoch(
			self,
			dataloader,
			criterion,
			optimizer,
	):
		batch_losses = []
		for x_batch, y_batch in dataloader:
			batch_loss = self._exec_batch(
				x_batch,
				y_batch,
				criterion,
				optimizer,
			)
			batch_losses.append(batch_loss)
		return np.mean(batch_losses)

	def _exec_batch(
			self,
			x_batch,
			y_batch,
			criterion,
			optimizer,
	):
		if self.training:
			log_p_y, out, h_sates = self.get_prediction_log_proba(
				x_batch, re_outputs_trace=True, re_hidden_states=True
			)
		else:
			with torch.no_grad():
				log_p_y, out, h_sates = self.get_prediction_log_proba(
					x_batch, re_outputs_trace=True, re_hidden_states=True
				)

		# TODO: add regularization loss
		# reg_loss = torch.mean(self.get_spikes_count_per_neuron(h_sates))
		# spikes = [h[-1] for l_name, h_list in h_sates.items() for h in h_list if l_name.lower() != "readout"]
		# reg_loss = 1e-5 * sum([torch.sum(s) for s in spikes])  # L1 loss on total number of spikes
		# reg_loss = 1e-5 * sum(
		# 	[torch.mean(torch.sum(torch.sum(s, dim=0), dim=0) ** 2) for s in spikes]
		# )  # L2 loss on spikes per neuron
		# reg_loss = torch.mean(self.get_spikes_count_per_neuron(h_sates) ** 2)

		batch_loss = criterion(log_p_y, y_batch.long().to(self.device))
		if self.training:
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()
		return batch_loss.item()

	def plot_loss_history(self, loss_history: LossHistory = None, show=False):
		if loss_history is None:
			loss_history = self.loss_history
		save_path = f"./{self.checkpoint_folder}/loss_history.png"
		os.makedirs(f"./{self.checkpoint_folder}/", exist_ok=True)
		loss_history.plot(save_path, show)

	def _create_checkpoint_path(self, epoch: int = -1):
		return f"./{self.checkpoint_folder}/{self.model_name}{SequentialModel.SUFFIX_SEP}{SequentialModel.CHECKPOINT_EPOCH_KEY}{epoch}{SequentialModel.SAVE_EXT}"

	def _create_new_checkpoint_meta(self, epoch: int, best: bool = False) -> dict:
		save_path = self._create_checkpoint_path(epoch)
		new_info = {SequentialModel.CHECKPOINT_EPOCHS_KEY: {epoch: save_path}}
		if best:
			new_info[SequentialModel.CHECKPOINT_BEST_KEY] = save_path
		return new_info

	def save_checkpoint(
			self,
			optimizer,
			epoch: int,
			epoch_losses: Dict[str, Any],
			best: bool = False,
	):
		os.makedirs(self.checkpoint_folder, exist_ok=True)
		save_path = self._create_checkpoint_path(epoch)
		torch.save({
			SequentialModel.CHECKPOINT_EPOCH_KEY: epoch,
			SequentialModel.CHECKPOINT_STATE_DICT_KEY: self.state_dict(),
			SequentialModel.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: optimizer.state_dict(),
			SequentialModel.CHECKPOINT_LOSS_KEY: epoch_losses,
		}, save_path)
		self.save_checkpoints_meta(self._create_new_checkpoint_meta(epoch, best))

	@staticmethod
	def get_save_path_from_checkpoints(
			checkpoints_meta: Dict[str, Union[str, Dict[Any, str]]],
			load_checkpoint_mode: LoadCheckpointMode = LoadCheckpointMode.BEST_EPOCH
	) -> str:
		if load_checkpoint_mode == load_checkpoint_mode.BEST_EPOCH:
			return checkpoints_meta[SequentialModel.CHECKPOINT_BEST_KEY]
		elif load_checkpoint_mode == load_checkpoint_mode.LAST_EPOCH:
			epochs_dict = checkpoints_meta[SequentialModel.CHECKPOINT_EPOCHS_KEY]
			last_epoch: int = max([int(e) for e in epochs_dict])
			return checkpoints_meta[SequentialModel.CHECKPOINT_EPOCHS_KEY][str(last_epoch)]
		else:
			raise ValueError()

	def get_checkpoints_loss_history(self) -> LossHistory:
		history = LossHistory()
		with open(self.checkpoints_meta_path, "r+") as jsonFile:
			meta: dict = json.load(jsonFile)
		checkpoints = [torch.load(path) for path in meta[SequentialModel.CHECKPOINT_EPOCHS_KEY].values()]
		for checkpoint in checkpoints:
			history.concat(checkpoint[SequentialModel.CHECKPOINT_LOSS_KEY])
		return history

	def load_checkpoint(
			self,
			load_checkpoint_mode: LoadCheckpointMode = LoadCheckpointMode.BEST_EPOCH
	) -> dict:
		with open(self.checkpoints_meta_path, "r+") as jsonFile:
			info: dict = json.load(jsonFile)
		path = self.get_save_path_from_checkpoints(info, load_checkpoint_mode)
		checkpoint = torch.load(path)
		self.load_state_dict(checkpoint[SequentialModel.CHECKPOINT_STATE_DICT_KEY], strict=True)
		return checkpoint

	def to_onnx(self, in_viz=None):
		if in_viz is None:
			in_viz = torch.randn((1, self.input_size), device=self.device)
		torch.onnx.export(
			self,
			in_viz,
			f"{self.checkpoint_folder}/{self.model_name}.onnx",
			verbose=True,
			input_names=None,
			output_names=None,
			opset_version=11
		)

	def save_checkpoints_meta(self, new_info: dict):
		info = dict()
		if os.path.exists(self.checkpoints_meta_path):
			with open(self.checkpoints_meta_path, "r+") as jsonFile:
				info = json.load(jsonFile)
		mapping_update_recursively(info, new_info)
		with open(self.checkpoints_meta_path, "w+") as jsonFile:
			json.dump(info, jsonFile, indent=4)

	def compute_classification_accuracy(
			self,
			dataloader: DataLoader,
			verbose: bool = False,
			desc: Optional[str] = None,
	) -> float:
		""" Computes classification accuracy on supplied data in batches. """
		self.eval()
		accs = []
		with torch.no_grad():
			for i, (inputs, classes) in tqdm(
					enumerate(dataloader), total=len(dataloader), desc=desc, disable=not verbose
			):
				inputs = inputs.to(self.device)
				classes = classes.to(self.device)
				outputs = self.get_prediction_logits(inputs, re_outputs_trace=False, re_hidden_states=False)
				_, preds = torch.max(outputs, -1)
				accs.extend(torch.eq(preds, classes).float().cpu().numpy())
		return np.mean(np.asarray(accs)).item()

	def compute_confusion_matrix(
			self,
			nb_classes: int,
			dataloaders: Dict[str, DataLoader],
			fit=False,
			fit_kwargs=None,
			load_checkpoint_mode: LoadCheckpointMode = None,
	):
		if fit_kwargs is None:
			fit_kwargs = {}
		if fit:
			self.fit(dataloaders['train'], dataloaders['val'], **fit_kwargs)

		if load_checkpoint_mode is not None:
			self.load_checkpoint(load_checkpoint_mode)
		return {key: self._compute_single_confusion_matrix(nb_classes, d) for key, d in dataloaders.items()}

	def _compute_single_confusion_matrix(self, nb_classes: int, dataloader: DataLoader) -> np.ndarray:
		self.eval()
		confusion_matrix = np.zeros((nb_classes, nb_classes))
		with torch.no_grad():
			for i, (inputs, classes) in enumerate(dataloader):
				inputs = inputs.to(self.device)
				classes = classes.to(self.device)
				outputs = self.get_prediction_logits(inputs, re_outputs_trace=False, re_hidden_states=False)
				_, preds = torch.max(outputs, -1)
				for t, p in zip(classes.view(-1), preds.view(-1)):
					confusion_matrix[t.long(), p.long()] += 1
		return confusion_matrix


