import json
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import torch
from torch import nn
from torchvision.transforms import Compose, Lambda
import torch.nn.functional as F

from ..callbacks import CheckpointManager, LoadCheckpointMode
from ..transforms import to_tensor


class BaseModel(torch.nn.Module):
	def __init__(
			self,
			input_sizes: Union[Dict[str, int], List[int], int],
			output_size: Union[Dict[str, int], List[int], int],
			name: str = "BaseModel",
			checkpoint_folder: str = "checkpoints",
			device: torch.device = None,
			input_transform: Union[Dict[str, Callable], List[Callable]] = None,
			**kwargs
	):
		super(BaseModel, self).__init__()
		self._input_sizes = BaseModel._format_sizes(input_sizes)
		self._output_size = BaseModel._format_sizes(output_size)
		self.name = name
		self.checkpoint_folder = checkpoint_folder
		self.kwargs = kwargs
		self.device = device
		if self.device is None:
			self._set_default_device_()

		self.input_transform: Dict[str, Callable] = self._make_input_transform(input_transform)
		self._add_to_device_transform_()

	@property
	def input_sizes(self) -> Dict[str, int]:
		return self._input_sizes

	@property
	def output_sizes(self) -> Dict[str, int]:
		return self._output_size

	@property
	def checkpoints_meta_path(self) -> str:
		full_filename = (
			f"{self.name}{CheckpointManager.SUFFIX_SEP}{CheckpointManager.CHECKPOINTS_META_SUFFIX}"
		)
		return f"{self.checkpoint_folder}/{full_filename}.json"

	@staticmethod
	def _format_sizes(sizes: Union[Dict[str, int], List[int], int]) -> Dict[str, int]:
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
			load_checkpoint_mode: LoadCheckpointMode = LoadCheckpointMode.BEST_ITR
	) -> dict:
		if checkpoints_meta_path is None:
			checkpoints_meta_path = self.checkpoints_meta_path
		with open(checkpoints_meta_path, "r+") as jsonFile:
			info: dict = json.load(jsonFile)
		path = CheckpointManager.get_save_name_from_checkpoints(info, load_checkpoint_mode)
		checkpoint = torch.load(path)
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

	def forward(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, torch.Tensor]:
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
