import enum
import json
import os
import shutil
import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from .base_callback import BaseCallback
from ..utils import mapping_update_recursively


class LoadCheckpointMode(enum.Enum):
	BEST_ITR = 0
	LAST_ITR = 1


class CheckpointManager(BaseCallback):
	"""
	This class is used to manage and create the checkpoints of a model.

	Attributes:
		checkpoint_folder (str): The folder to save the checkpoints to.
		meta_path_prefix (str): The prefix to use for the checkpoint's metadata file.
		metric (str): The name of the metric to collect the best checkpoint on.
		minimise_metric (bool): Whether to minimise the metric or maximise it.
		curr_best_metric (float): The current best metric value.
	"""

	SAVE_EXT = '.pth'
	SUFFIX_SEP = '-'
	CHECKPOINTS_META_SUFFIX = 'checkpoints'
	CHECKPOINT_SAVE_PATH_KEY = "save_path"
	CHECKPOINT_BEST_KEY = "best"
	CHECKPOINT_ITRS_KEY = "iterations"
	CHECKPOINT_ITR_KEY = "itr"
	CHECKPOINT_METRICS_KEY = 'rewards'
	CHECKPOINT_OPTIMIZER_STATE_DICT_KEY = "optimizer_state_dict"
	CHECKPOINT_STATE_DICT_KEY = "model_state_dict"
	CHECKPOINT_TRAINING_HISTORY_KEY = "training_history"
	CHECKPOINT_FILE_STRUCT: Dict[str, Union[str, Dict[int, str]]] = {
		CHECKPOINT_BEST_KEY: CHECKPOINT_SAVE_PATH_KEY,
		CHECKPOINT_ITRS_KEY: {0: CHECKPOINT_SAVE_PATH_KEY},
	}
	load_mode_to_suffix = {mode: mode.name for mode in list(LoadCheckpointMode)}

	@staticmethod
	def _replace_trainer_history(trainer, new_history: Any):
		"""
		Replaces the training history in the trainer with the given history.
		:param trainer: The trainer object.
		:param new_history: The new history object.
		:return: None
		"""
		trainer.callbacks.remove(trainer.training_history)
		trainer.callbacks.append(new_history)
		trainer.training_history = new_history
		trainer.sort_callbacks_()

	@staticmethod
	def get_save_name_from_checkpoints(
			checkpoints_meta: Dict[str, Union[str, Dict[Any, str]]],
			load_checkpoint_mode: LoadCheckpointMode = LoadCheckpointMode.BEST_ITR
	) -> str:
		"""
		Gets the save name from the checkpoint's metadata given the load checkpoint mode.
		:param checkpoints_meta: The checkpoint's metadata.
		:param load_checkpoint_mode: The load checkpoint mode.
		:return: The save name.
		"""
		if load_checkpoint_mode == load_checkpoint_mode.BEST_ITR:
			if CheckpointManager.CHECKPOINT_BEST_KEY in checkpoints_meta:
				return checkpoints_meta[CheckpointManager.CHECKPOINT_BEST_KEY]
			else:
				raise FileNotFoundError(
					f"No best checkpoint found in checkpoints_meta. "
					f"Please use a different load_checkpoint_mode."
				)
		elif load_checkpoint_mode == load_checkpoint_mode.LAST_ITR:
			itr_dict = checkpoints_meta[CheckpointManager.CHECKPOINT_ITRS_KEY]
			last_itr: int = max([int(e) for e in itr_dict])
			return checkpoints_meta[CheckpointManager.CHECKPOINT_ITRS_KEY][str(last_itr)]
		else:
			raise ValueError("Invalid load_checkpoint_mode")

	def __init__(
			self,
			checkpoint_folder: str = "./checkpoints",
			*,
			meta_path_prefix: str = "network",
			metric: str = "val_loss",
			minimise_metric: bool = True,
			verbose: bool = False
	):
		"""
		Initialises the checkpoint manager.
		:param checkpoint_folder: The folder to save the checkpoints to.
		:param meta_path_prefix: The prefix to use for the checkpoint's metadata file.
		:param metric: The name of the metric to collect the best checkpoint on.
		:param minimise_metric: Whether to minimise the metric or maximise it.
		:param verbose: Whether to print out the trace of the checkpoint manager.
		"""
		self.checkpoint_folder = checkpoint_folder
		self.meta_path_prefix = meta_path_prefix
		self.verbose = verbose

		self.metric = metric
		self.minimise_metric = minimise_metric
		self.curr_best_metric = np.inf if self.minimise_metric else -np.inf

	@property
	def checkpoints_meta_path(self) -> str:
		"""
		Gets the path to the checkpoints metadata file.
		:return: The path to the checkpoints metadata file.
		"""
		full_filename = (
			f"{self.meta_path_prefix}{CheckpointManager.SUFFIX_SEP}{CheckpointManager.CHECKPOINTS_META_SUFFIX}"
		)
		return f"{self.checkpoint_folder}/{full_filename}.json"

	def get_checkpoint_filename(self, itr: int = -1):
		"""
		Generate the filename for the checkpoint at the given iteration.
		:param itr: The iteration to generate the filename for.
		:return: The filename for the checkpoint at the given iteration.
		"""
		pre_name = f"{self.meta_path_prefix}"
		if itr == -1:
			post_name = ""
		else:
			post_name = f"{CheckpointManager.SUFFIX_SEP}{CheckpointManager.CHECKPOINT_ITR_KEY}{itr}"
		return f"{pre_name}{post_name}{CheckpointManager.SAVE_EXT}"

	def _create_new_checkpoint_meta(self, itr: int, best: bool = False) -> dict:
		"""
		Creates a new checkpoint's metadata.
		:param itr: The iteration of the checkpoint.
		:param best: Whether the checkpoint is currently the best.
		:return: The new checkpoint's metadata.
		"""
		save_name = self.get_checkpoint_filename(itr)
		new_info = {CheckpointManager.CHECKPOINT_ITRS_KEY: {str(itr): save_name}}
		if best:
			new_info[CheckpointManager.CHECKPOINT_BEST_KEY] = save_name
		return new_info

	def save_checkpoint(
			self,
			itr: int,
			itr_metrics: Dict[str, Any],
			best: bool = False,
			state_dict: Optional[Dict[str, Any]] = None,
			optimizer_state_dict: Optional[Dict[str, Any]] = None,
			training_history: Optional[Any] = None,
	) -> str:
		"""
		Saves a checkpoint of the model and optimizer states at the given iteration.
		:param itr: The iteration number.
		:param itr_metrics: The metrics at the given iteration.
		:param best: Whether this is the best iteration so far.
		:param state_dict: The state dict of the model.
		:param optimizer_state_dict: The state dict of the optimizer.
		:param training_history: The training history object.
		:return: The path to the saved checkpoint.
		"""
		os.makedirs(self.checkpoint_folder, exist_ok=True)
		save_name = self.get_checkpoint_filename(itr)
		path = os.path.join(self.checkpoint_folder, save_name)
		torch.save({
			CheckpointManager.CHECKPOINT_ITR_KEY: itr,
			CheckpointManager.CHECKPOINT_STATE_DICT_KEY: state_dict,
			CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: optimizer_state_dict,
			CheckpointManager.CHECKPOINT_METRICS_KEY: itr_metrics,
			CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY: training_history,
		}, path)
		self.save_checkpoints_meta(self._create_new_checkpoint_meta(itr, best))
		return path

	def load_checkpoint(
			self,
			load_checkpoint_mode: LoadCheckpointMode = LoadCheckpointMode.BEST_ITR
	) -> dict:
		"""
		Loads the checkpoint at the given load_checkpoint_mode.
		:param load_checkpoint_mode:
		:return:
		"""
		with open(self.checkpoints_meta_path, "r+") as jsonFile:
			info: dict = json.load(jsonFile)
		filename = CheckpointManager.get_save_name_from_checkpoints(info, load_checkpoint_mode)
		checkpoint = torch.load(f"{self.checkpoint_folder}/{filename}")
		return checkpoint

	def save_checkpoints_meta(self, new_info: dict):
		info = dict()
		if os.path.exists(self.checkpoints_meta_path):
			with open(self.checkpoints_meta_path, "r+") as jsonFile:
				info = json.load(jsonFile)
		mapping_update_recursively(info, new_info)
		os.makedirs(self.checkpoint_folder, exist_ok=True)
		with open(self.checkpoints_meta_path, "w+") as jsonFile:
			json.dump(info, jsonFile, indent=4)

	def start(self, trainer):
		start_itr = 0
		if trainer.load_checkpoint_mode is None:
			if os.path.exists(self.checkpoints_meta_path):
				if trainer.force_overwrite:
					shutil.rmtree(self.checkpoint_folder)
				else:
					raise ValueError(
						f"{self.checkpoints_meta_path} already exists. "
						f"Set force_overwrite flag to True to overwrite existing saves."
					)
		else:
			try:
				checkpoint = self.load_checkpoint(trainer.load_checkpoint_mode)
				trainer.model.load_state_dict(checkpoint[CheckpointManager.CHECKPOINT_STATE_DICT_KEY], strict=True)
				start_itr = int(checkpoint[CheckpointManager.CHECKPOINT_ITR_KEY]) + 1
				self._replace_trainer_history(trainer, checkpoint[CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY])
			except FileNotFoundError as e:
				if self.verbose:
					warnings.warn(f"Error: {e}", Warning)
					warnings.warn("No such checkpoint. Fit from beginning.")

		trainer.current_training_state = trainer.current_training_state.update(iteration=start_itr)
		if self.minimise_metric:
			self.curr_best_metric = trainer.training_history.min(self.metric)
		else:
			self.curr_best_metric = trainer.training_history.max(self.metric)

	def on_iteration_end(self, trainer):
		if self.minimise_metric:
			is_best = trainer.current_training_state.itr_metrics[self.metric] < self.curr_best_metric
		else:
			is_best = trainer.current_training_state.itr_metrics[self.metric] > self.curr_best_metric
		if is_best:
			self.curr_best_metric = trainer.current_training_state.itr_metrics[self.metric]
		self.save_checkpoint(
			trainer.current_training_state.iteration, trainer.current_training_state.itr_metrics, is_best,
			state_dict=trainer.model.state_dict(),
			optimizer_state_dict=trainer.optimizer.state_dict(),
			training_history=trainer.training_history,
		)
		if trainer.training_history:
			trainer.training_history.plot(
				save_path=os.path.join(self.checkpoint_folder, "training_history.png"),
				show=False
			)
