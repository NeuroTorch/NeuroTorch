import enum
import json
import logging
import os
import pprint
import shutil
import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from .base_callback import BaseCallback
from ..utils import mapping_update_recursively


class LoadCheckpointMode(enum.Enum):
    """
    Enum for the different modes of loading a checkpoint.

    :Attributes:
        - **BEST_ITR** (int): Indicates that the iteration with the best metric will be loaded.
        - **LAST_ITR** (int): Indicates that the last iteration will be loaded.
    """
    BEST_ITR = 0
    LAST_ITR = 1

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def from_str(mode_name: str) -> 'LoadCheckpointMode':
        """
        Converts a string to a :class:`LoadCheckpointMode` instance.

        :param mode_name: The name of the mode.
        :type mode_name: str
        :return: The corresponding :class:`LoadCheckpointMode` instance.
        :rtype: LoadCheckpointMode
        """
        if mode_name.lower() in ["best", "last"]:
            mode_name = mode_name.upper() + "_ITR"
        assert mode_name.upper() in LoadCheckpointMode.__members__, f"Invalid mode name: {mode_name}"
        return LoadCheckpointMode[mode_name.upper()]


class CheckpointManager(BaseCallback):
    """
    This class is used to manage and create the checkpoints of a model.

    :Attributes:
        - **checkpoint_folder** (str): The folder to save the checkpoints to.
        - **meta_path_prefix** (str): The prefix to use for the checkpoint's metadata file.
        - **metric** (str): The name of the metric to collect the best checkpoint on.
        - **minimise_metric** (bool): Whether to minimise the metric or maximise it.
        - **curr_best_metric** (float): The current best metric value.
    """

    DEFAULT_PRIORITY = BaseCallback.DEFAULT_LOW_PRIORITY

    SAVE_EXT: str = '.pth'
    SUFFIX_SEP: str = '-'
    CHECKPOINTS_META_SUFFIX: str = 'checkpoints'
    CHECKPOINT_SAVE_PATH_KEY: str = "save_path"
    CHECKPOINT_BEST_KEY: str = "best"
    CHECKPOINT_ITRS_KEY: str = "iterations"
    CHECKPOINT_ITR_KEY: str = "itr"
    CHECKPOINT_METRICS_KEY: str = 'metrics'
    CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: str = "optimizer_state_dict"
    CHECKPOINT_STATE_DICT_KEY: str = "model_state_dict"
    CHECKPOINT_TRAINING_HISTORY_KEY: str = "training_history"
    CHECKPOINT_FILE_STRUCT: Dict[str, Union[str, Dict[int, str]]] = {
        CHECKPOINT_BEST_KEY: CHECKPOINT_SAVE_PATH_KEY,
        CHECKPOINT_ITRS_KEY: {0: CHECKPOINT_SAVE_PATH_KEY},
    }
    load_mode_to_suffix: Dict[LoadCheckpointMode, str] = {mode: mode.name for mode in list(LoadCheckpointMode)}

    @staticmethod
    def _replace_trainer_history(trainer, new_history: Any):
        """
        Replaces the training history in the trainer with the given history.
        :param trainer: The trainer object.
        :param new_history: The new history object.
        :return: None
        """
        # trainer.callbacks.remove(trainer.training_history)
        # trainer.callbacks.append(new_history)
        # trainer.training_history = new_history
        trainer.training_history.load_checkpoint_state(trainer, new_history)
        trainer.sort_callbacks_()

    @staticmethod
    def get_save_name_from_checkpoints(
            checkpoints_meta: Dict[str, Union[str, Dict[Any, str]]],
            load_checkpoint_mode: LoadCheckpointMode = LoadCheckpointMode.BEST_ITR
    ) -> str:
        """
        Gets the save name from the checkpoint's metadata given the load checkpoint mode.

        :param checkpoints_meta: The checkpoint's metadata.
        :type checkpoints_meta: Dict[str, Union[str, Dict[Any, str]]]
        :param load_checkpoint_mode: The load checkpoint mode.
        :type load_checkpoint_mode: LoadCheckpointMode

        :return: The save name.
        :rtype: str
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
            checkpoints_meta_path: Optional[str] = None,
            meta_path_prefix: str = "network",
            metric: str = "val_loss",
            minimise_metric: bool = True,
            save_freq: int = 1,
            save_best_only: bool = False,
            start_save_at: int = 0,
            verbose: bool = False,
            **kwargs
    ):
        """
        Initialises the checkpoint manager.

        :param checkpoint_folder: The folder to save the checkpoints to.
        :type checkpoint_folder: str
        :param checkpoints_meta_path: The path to the checkpoints metadata file. If None, will use the
                                        `checkpoint_folder` and `meta_path_prefix` to create the path.
        :type checkpoints_meta_path: Optional[str]
        :param meta_path_prefix: The prefix to use for the checkpoint's metadata file.
        :type meta_path_prefix: str
        :param metric: The name of the metric to collect the best checkpoint on.
        :type metric: str
        :param minimise_metric: Whether to minimise the metric or maximise it.
        :type minimise_metric: bool
        :param save_freq: The frequency at which to save checkpoints. If set to <= 0, will save at the end of the
                            training.
        :type save_freq: int
        :param save_best_only: Whether to only save the best checkpoint.
        :type save_best_only: bool
        :param start_save_at: The iteration at which to start saving checkpoints.
        :type start_save_at: int
        :param verbose: Whether to print out the trace of the checkpoint manager.
        :type verbose: bool
        :param kwargs: The keyword arguments to pass to the BaseCallback.

        :keyword: **show_best_metric_on_p_bar** (bool): Whether to show the best metric on the progress bar.
            Default: `save_best_only`.
        """
        kwargs.setdefault("save_state", False)
        super().__init__(**kwargs)
        os.makedirs(checkpoint_folder, exist_ok=True)
        self.checkpoint_folder = checkpoint_folder
        self._checkpoints_meta_path = checkpoints_meta_path
        self.meta_path_prefix = meta_path_prefix
        self.verbose = verbose

        self.metric = metric
        self.minimise_metric = minimise_metric
        self.save_freq = save_freq
        self.save_best_only = save_best_only
        if self.save_best_only:
            self.save_freq = -1
        self.start_save_at = start_save_at
        self.curr_best_metric = np.inf if self.minimise_metric else -np.inf
        self.curr_checkpoint = None
        self.show_best_metric_on_p_bar = kwargs.get("show_best_metric_on_p_bar", self.save_best_only)

    @property
    def checkpoints_meta_path(self) -> str:
        """
        Gets the path to the checkpoints metadata file.

        :return: The path to the checkpoints metadata file.
        :rtype: str
        """
        if self._checkpoints_meta_path is not None:
            return self._checkpoints_meta_path
        full_filename = (
            f"{self.meta_path_prefix}{CheckpointManager.SUFFIX_SEP}{CheckpointManager.CHECKPOINTS_META_SUFFIX}"
        )
        return f"{self.checkpoint_folder}/{full_filename}.json"

    @checkpoints_meta_path.setter
    def checkpoints_meta_path(self, value: str):
        """
        Sets the path to the checkpoints metadata file.

        :param value: The path to the checkpoints metadata file.
        :type value: str

        :return: None
        """
        self._checkpoints_meta_path = value

    def get_checkpoint_filename(self, itr: int = -1):
        """
        Generate the filename for the checkpoint at the given iteration.

        :param itr: The iteration to generate the filename for.
        :type itr: int

        :return: The filename for the checkpoint at the given iteration.
        :rtype: str
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
        :type itr: int
        :param best: Whether the checkpoint is currently the best.
        :type best: bool

        :return: The new checkpoint's metadata.
        :rtype: dict
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
            **other_states,
    ) -> str:
        """
        Saves a checkpoint of the model and optimizer states at the given iteration.

        :param itr: The iteration number.
        :type itr: int
        :param itr_metrics: The metrics at the given iteration.
        :type itr_metrics: Dict[str, Any]
        :param best: Whether this is the best iteration so far.
        :type best: bool
        :param state_dict: The state dict of the model.
        :type state_dict: Optional[Dict[str, Any]]
        :param optimizer_state_dict: The state dict of the optimizer.
        :type optimizer_state_dict: Optional[Dict[str, Any]]
        :param training_history: The training history object.
        :type training_history: Optional[Any]

        :return: The path to the saved checkpoint.
        :rtype: str
        """
        os.makedirs(self.checkpoint_folder, exist_ok=True)
        save_name = self.get_checkpoint_filename(itr)
        path = os.path.join(self.checkpoint_folder, save_name)
        basic_states = {
            CheckpointManager.CHECKPOINT_ITR_KEY: itr,
            CheckpointManager.CHECKPOINT_STATE_DICT_KEY: state_dict,
            CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: optimizer_state_dict,
            CheckpointManager.CHECKPOINT_METRICS_KEY: itr_metrics,
            CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY: training_history,
        }
        assert all(key not in other_states for key in basic_states.keys()), (
            f"Other states cannot have the same keys as the basic states ({basic_states.keys()})."
        )
        states = {**basic_states, **other_states}
        torch.save(states, path)
        self.save_checkpoints_meta(self._create_new_checkpoint_meta(itr, best))
        return path

    def load_checkpoint(
            self,
            load_checkpoint_mode: LoadCheckpointMode = LoadCheckpointMode.BEST_ITR
    ) -> dict:
        """
        Loads the checkpoint at the given load_checkpoint_mode.

        :param load_checkpoint_mode: The load_checkpoint_mode to use.
        :type load_checkpoint_mode: LoadCheckpointMode

        :return: The loaded checkpoint.
        :rtype: dict
        """
        # TODO: add the possibility to load a specific itr
        with open(self.checkpoints_meta_path, "r+") as jsonFile:
            info: dict = json.load(jsonFile)
        filename = CheckpointManager.get_save_name_from_checkpoints(info, load_checkpoint_mode)
        checkpoint = torch.load(f"{self.checkpoint_folder}/{filename}")
        return checkpoint

    def save_checkpoints_meta(self, new_info: dict):
        """
        Saves the new checkpoints' metadata.

        :param new_info: The new checkpoints' metadata.
        :type new_info: dict

        :return: None
        """
        info = dict()
        if os.path.exists(self.checkpoints_meta_path):
            with open(self.checkpoints_meta_path, "r+") as jsonFile:
                info = json.load(jsonFile)
        mapping_update_recursively(info, new_info)
        os.makedirs(self.checkpoint_folder, exist_ok=True)
        with open(self.checkpoints_meta_path, "w+") as jsonFile:
            json.dump(info, jsonFile, indent=4)

    def start(self, trainer, **kwargs):
        """
        Call at the beginning of the training by the Trainer. Load the checkpoint base on the load_checkpoint_mode of
        the trainer and update the current_training_state of the trainer.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        super().start(trainer)
        if trainer.load_checkpoint_mode is None:
            trainer.load_checkpoint_mode = LoadCheckpointMode.LAST_ITR
        start_itr = 0
        checkpoint = self.curr_checkpoint
        if trainer.force_overwrite:
            if os.path.exists(self.checkpoints_meta_path):
                shutil.rmtree(self.checkpoint_folder)
        else:
            try:
                checkpoint = self.load_checkpoint(trainer.load_checkpoint_mode)
                trainer.model.load_state_dict(checkpoint[CheckpointManager.CHECKPOINT_STATE_DICT_KEY], strict=True)
                if trainer.optimizer is not None:
                    trainer.optimizer.load_state_dict(checkpoint[CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY])
                start_itr = int(checkpoint[CheckpointManager.CHECKPOINT_ITR_KEY]) + 1
            # self._replace_trainer_history(trainer, checkpoint[CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY])
            except FileNotFoundError as e:
                if self.verbose:
                    logging.info("No such checkpoint. Fit from beginning.")
            finally:
                self.curr_checkpoint = checkpoint

        trainer.update_state_(iteration=start_itr)
        if self.minimise_metric:
            self.curr_best_metric = trainer.training_history.min(self.metric)
        else:
            self.curr_best_metric = trainer.training_history.max(self.metric)

    def save_on(self, trainer) -> bool:
        """
        Saves the checkpoint if the current iteration is a checkpoint iteration.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: Whether the checkpoint was saved.
        :rtype: bool
        """
        itr_metrics = trainer.current_training_state.itr_metrics
        if itr_metrics is None:
            itr_metrics = {}
        other_states = trainer.callbacks.get_checkpoint_state(trainer)
        is_best = self._check_is_best(trainer)
        if is_best:
            self.curr_best_metric = itr_metrics[self.metric]
        self.save_checkpoint(
            trainer.current_training_state.iteration, itr_metrics, is_best,
            state_dict=trainer.model.state_dict(),
            optimizer_state_dict=trainer.optimizer.state_dict() if trainer.optimizer else None,
            training_history=trainer.training_history.get_checkpoint_state(trainer),
            **other_states
        )
        if trainer.training_history:
            trainer.training_history.plot(
                save_path=os.path.join(self.checkpoint_folder, "training_history.png"),
                show=False
            )
        return True

    def on_iteration_end(self, trainer, **kwargs):
        """
        Called when an iteration ends. An iteration is defined as one full pass through the training dataset and
        the validation dataset. The checkpoint is saved if the current constraints are met.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        if self.start_save_at > trainer.current_training_state.iteration:
            return

        if self.save_best_only:
            if self._check_is_best(trainer):
                self.save_on(trainer)
            return

        if self.save_freq > 0 and trainer.current_training_state.iteration % self.save_freq == 0:
            return self.save_on(trainer)
        if trainer.current_training_state.iteration >= trainer.state.n_iterations - 1:
            return self.save_on(trainer)

    def _check_is_best(self, trainer) -> Optional[bool]:
        if trainer.current_training_state.itr_metrics is None:
            return None

        itr_metric = trainer.current_training_state.itr_metrics.get(self.metric, self.curr_best_metric)
        if self.minimise_metric:
            is_best = itr_metric < self.curr_best_metric
        else:
            is_best = itr_metric > self.curr_best_metric
        return is_best

    def close(self, trainer, **kwargs):
        """
        Called when the training is finished. Saves the current checkpoint if the current iteration is lower than
        the number of iterations i.e. there is new stuff to save.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        if trainer.current_training_state.iteration < trainer.state.n_iterations:
            self.save_on(trainer)

    def extra_repr(self) -> str:
        extra_repr = f"folder={self.checkpoint_folder}, metric={self.metric}, minimise={self.minimise_metric}"
        extra_repr += f", start_save_at={self.start_save_at}"
        extra_repr += f", save_freq={self.save_freq}"
        extra_repr += f", save_best_only={self.save_best_only}"
        return extra_repr

    def on_pbar_update(self, trainer, **kwargs) -> dict:
        """
        Called when the progress bar is updated. Adds the current best metric to the progress bar.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: The dictionary to add to the progress bar.
        :rtype: dict
        """
        if self.show_best_metric_on_p_bar:
            return {f"best_{self.metric}": self.curr_best_metric}
        return {}
