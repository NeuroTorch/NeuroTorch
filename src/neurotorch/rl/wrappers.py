from typing import Any, Optional

import numpy as np
import torch
from mlagents_envs.base_env import ActionTuple


class TensorActionTuple:
	def __init__(
			self,
			continuous: Optional[torch.Tensor] = None,
			discrete: Optional[torch.Tensor] = None
	):
		self._continuous = self._cast_entry(continuous)
		self._discrete = self._cast_entry(discrete)
		self.add_empty()

	@property
	def continuous(self) -> Optional[torch.Tensor]:
		return self._continuous

	@property
	def discrete(self) -> Optional[torch.Tensor]:
		return self._discrete

	@staticmethod
	def _cast_entry(entry: Any) -> Optional[torch.Tensor]:
		if entry is None:
			return None
		if isinstance(entry, np.ndarray):
			entry = torch.from_numpy(entry)
		elif isinstance(entry, torch.Tensor):
			pass
		else:
			entry = torch.tensor(entry, dtype=torch.float32)
		return entry if torch.numel(entry) > 0 else None

	def add_empty(self) -> None:
		assert (self._continuous is not None) or (self._discrete is not None), "TensorActionTuple is empty"
		if self._discrete is None:
			self._discrete = torch.empty((self._continuous.shape[0], 0), dtype=torch.long)
		if self._continuous is None:
			self._continuous = torch.empty((self._discrete.shape[0], 0), dtype=torch.float32)

	@staticmethod
	def from_numpy(action_tuple: ActionTuple) -> 'TensorActionTuple':
		return TensorActionTuple(torch.from_numpy(action_tuple.continuous), torch.from_numpy(action_tuple.discrete))

	def to_numpy(self) -> ActionTuple:
		if self._continuous is None:
			continuous_outputs = None
		else:
			continuous_outputs = self._continuous.detach().cpu().numpy()

		if self._discrete is None:
			discrete_outputs = None
		else:
			discrete_outputs = self._discrete.detach().cpu().numpy()
		return ActionTuple(continuous_outputs, discrete_outputs)

	def copy(self) -> 'TensorActionTuple':
		return TensorActionTuple(
			self._continuous.detach().clone() if self._continuous is not None else None,
			self._discrete.detach().clone() if self._discrete is not None else None
		)



