import json
import logging
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import torch
from torch import nn
from torchvision.transforms import Compose, Lambda
import torch.nn.functional as F

from ..callbacks import CheckpointManager, LoadCheckpointMode
from ..dimension import DimensionLike, SizeTypes
from ..transforms import to_tensor


class BaseModel(torch.nn.Module):
	def __init__(
			self,
			input_sizes: Optional[Union[Dict[str, DimensionLike], SizeTypes]] = None,
			output_size: Optional[Union[Dict[str, DimensionLike], SizeTypes]] = None,
			name: str = "BaseModel",
			checkpoint_folder: str = "checkpoints",
			device: torch.device = None,
			input_transform: Union[Dict[str, Callable], List[Callable]] = None,
			**kwargs
	):
		super(BaseModel, self).__init__()
		self._is_built = False
		self._given_input_transform = input_transform
		self.input_transform: Dict[str, Callable] = None
		self.input_sizes = input_sizes
		self.output_sizes = output_size
		self.name = name
		self.checkpoint_folder = checkpoint_folder
		self.kwargs = kwargs
		self.device = device
		if self.device is None:
			self._set_default_device_()

	@property
	def input_sizes(self) -> Dict[str, int]:
		return self._input_sizes

	@input_sizes.setter
	def input_sizes(self, input_sizes: Union[Dict[str, DimensionLike], SizeTypes]):
		# if self.input_sizes is not None:
		# 	raise ValueError("Input sizes can only be set once.")
		if input_sizes is not None:
			self._input_sizes = self._format_sizes(input_sizes)
			self.input_transform: Dict[str, Callable] = self._make_input_transform(self._given_input_transform)
			self._add_to_device_transform_()

	@property
	def output_sizes(self) -> Dict[str, int]:
		return self._output_sizes

	@output_sizes.setter
	def output_sizes(self, output_size: Union[Dict[str, DimensionLike], SizeTypes]):
		# if self._output_sizes is not None:
		# 	raise ValueError("Output sizes can only be set once.")
		if output_size is not None:
			self._output_sizes = self._format_sizes(output_size)

	@property
	def _ready(self):
		is_all_not_none = all([s is not None for s in [self._input_sizes, self._output_sizes]])
		if is_all_not_none:
			is_any_none = any([s is None for s in list(self._input_sizes.values()) + list(self._output_sizes.values())])
		else:
			is_any_none = True
		return is_all_not_none and not is_any_none

	@property
	def is_built(self) -> bool:
		return self._is_built

	@property
	def checkpoints_meta_path(self) -> str:
		full_filename = (
			f"{self.name}{CheckpointManager.SUFFIX_SEP}{CheckpointManager.CHECKPOINTS_META_SUFFIX}"
		)
		return f"{self.checkpoint_folder}/{full_filename}.json"

	@staticmethod
	def _format_sizes(sizes: Union[Dict[str, DimensionLike], SizeTypes]) -> Dict[str, int]:
		if isinstance(sizes, dict):
			return sizes
		elif isinstance(sizes, list):
			return {
				f"{i}": s
				for i, s in enumerate(sizes)
			}
		else:
			return {
				"0": sizes
			}

	def _make_input_transform(self, input_transform: Union[Dict[str, Callable], List[Callable]]) -> Dict[str, Callable]:
		if input_transform is None:
			input_transform = self.get_default_transform()
		if isinstance(input_transform, list):
			default_transform = self.get_default_transform()
			if len(input_transform) < len(self.input_sizes):
				for i in range(len(input_transform), len(self.input_sizes)):
					input_transform.append(default_transform[list(self.input_sizes.keys())[i]])
			input_transform = {in_name: t for in_name, t in zip(self.input_sizes, input_transform)}
		if isinstance(input_transform, dict):
			assert all([in_name in input_transform for in_name in self.input_sizes]), \
				f"Input transform must contain all input names: {self.input_sizes.keys()}"
		return input_transform

	def load_checkpoint(
			self,
			checkpoints_meta_path: Optional[str] = None,
			load_checkpoint_mode: LoadCheckpointMode = LoadCheckpointMode.BEST_ITR,
			verbose: bool = True
	) -> dict:
		if checkpoints_meta_path is None:
			checkpoints_meta_path = self.checkpoints_meta_path
		with open(checkpoints_meta_path, "r+") as jsonFile:
			info: dict = json.load(jsonFile)
		save_name = CheckpointManager.get_save_name_from_checkpoints(info, load_checkpoint_mode)
		checkpoint_path = f"{self.checkpoint_folder}/{save_name}"
		if verbose:
			logging.info(f"Loading checkpoint from {checkpoint_path}")
		checkpoint = torch.load(checkpoint_path, map_location=self.device)
		self.load_state_dict(checkpoint[CheckpointManager.CHECKPOINT_STATE_DICT_KEY], strict=True)
		return checkpoint

	def get_default_transform(self) -> Dict[str, nn.Module]:
		return {
			in_name: Compose([
				Lambda(lambda a: to_tensor(a, dtype=torch.float32)),
			])
			for in_name in self.input_sizes
		}

	def apply_transform(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
		"""
		:param inputs: dict of inputs of shape (batch_size, *input_size)
		:return: The input of the network with the same shape as the input.
		"""
		assert all([in_name in self.input_sizes for in_name in inputs]), \
			f"Inputs must be all in input names: {self.input_sizes.keys()}"
		inputs = {
			in_name: torch.stack(
				[self.input_transform[in_name](obs_i) for obs_i in in_batch],
				dim=0
			)
			for in_name, in_batch in inputs.items()
		}
		return inputs

	def _add_to_device_transform_(self):
		for in_name, trans in self.input_transform.items():
			self.input_transform[in_name] = Compose([trans, Lambda(lambda t: t.to(self.device))])

	def _set_default_device_(self):
		self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	def infer_sizes_from_inputs(self, inputs: Union[Dict[str, Any], torch.Tensor]):
		if isinstance(inputs, torch.Tensor):
			inputs = {
				"0": inputs
			}
		self.input_sizes = {k: v.shape[1:] for k, v in inputs.items()}

	def build(self, *args, **kwargs):
		"""
		Build the network.
		:param args:
		:param kwargs:
		:return:
		"""
		self._is_built = True

	def __call__(self, inputs: Union[Dict[str, Any], torch.Tensor], *args, **kwargs):
		if not self._is_built:
			self.infer_sizes_from_inputs(inputs)
			self.build()
		return super(BaseModel, self).__call__(inputs, *args, **kwargs)

	def forward(self, inputs: Union[Dict[str, Any], torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
		raise NotImplementedError()

	def get_prediction_trace(
			self,
			inputs: Union[Dict[str, Any], torch.Tensor],
			**kwargs
	) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
		raise NotImplementedError()
	
	def get_raw_prediction(
			self,
			inputs: torch.Tensor,
			re_outputs_trace: bool = True,
			re_hidden_states: bool = True
	) -> Union[Tuple[Any, Any, Any], Tuple[Any, Any], Any]:
		raise NotImplementedError()
	
	def get_prediction_proba(
			self,
			inputs: torch.Tensor,
			re_outputs_trace: bool = True,
			re_hidden_states: bool = True
	) -> Union[Tuple[Any, Any, Any], Tuple[Any, Any], Any]:
		m, *outs = self.get_raw_prediction(inputs, re_outputs_trace, re_hidden_states)
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
			return proba, *outs
		return proba
	
	def get_prediction_log_proba(
			self,
			inputs: torch.Tensor,
			re_outputs_trace: bool = True,
			re_hidden_states: bool = True
	) -> Union[Tuple[Any, Any, Any], Tuple[Any, Any], Any]:
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

	def soft_update(self, other: 'BaseModel', tau: float = 1e-2) -> None:
		"""
		Copies the weights from the other network to this network with a factor of tau
		"""
		with torch.no_grad():
			for param, other_param in zip(self.parameters(), other.parameters()):
				param.data.copy_((1 - tau) * param.data + tau * other_param.data)

	def hard_update(self, other: 'BaseModel') -> None:
		"""
		Copies the weights from the other network to this network
		"""
		with torch.no_grad():
			self.load_state_dict(other.state_dict())

	def to_onnx(self, in_viz=None):
		if in_viz is None:
			in_viz = torch.randn((1, self.input_sizes), device=self.device)
		torch.onnx.export(
			self,
			in_viz,
			f"{self.checkpoint_folder}/{self.name}.onnx",
			verbose=True,
			input_names=None,
			output_names=None,
			opset_version=11
		)
