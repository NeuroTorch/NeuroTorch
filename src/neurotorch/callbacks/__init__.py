import numbers

import numpy as np

from .base_callback import (
    BaseCallback,
    CallbacksList,
)

from .checkpoints_manager import (
    LoadCheckpointMode,
    CheckpointManager
)

from .history import (
    TrainingHistory,
)

from . import early_stopping


class ForesightTimeStepUpdaterOnTarget(BaseCallback):
    """
    Updates the foresight time step of the model to the length of the target sequence. This is useful for models that
    are trained with a fixed foresight time step, but are evaluated on sequences of different lengths.
    """
    DEFAULT_PRIORITY = BaseCallback.DEFAULT_HIGH_PRIORITY

    def __init__(self, **kwargs):
        """
        Constructor for ForesightTimeStepUpdaterOnTarget class.

        :keyword int time_steps_multiplier: The multiplier to use to determine the time steps. Defaults to 1.
        :keyword bool target_skip_first: Whether to skip the first time step of the target sequence. Defaults to True.
        :keyword int update_val_loss_freq: The frequency at which to update the validation loss. Defaults to 1.
        :keyword float start_intensive_val_at: The fraction of the training epochs at which to start intensive validation.
            An intensive validation is when the validation is performed at each iteration. Defaults to 0.0.
            If the value is an integer, it is interpreted as the number of iterations at which to start intensive
            validation.
        :keyword str hh_memory_size_strategy: The strategy to use to determine the hidden history memory size.
            The available strategies are:
                - "out_memory_size": The hidden history memory size is equal to the output memory size.
                - "foresight_time_steps": The hidden history memory size is equal to the foresight time steps.
                - <Number>: The hidden history memory size is equal to the specified number.
            Defaults to "out_memory_size".
        """
        kwargs["save_state"] = False
        super().__init__(**kwargs)
        self._hh_memory_size_cache = None
        self._out_memory_size_cache = None
        self._foresight_time_steps_cache = None
        self.time_steps_multiplier = kwargs.get("time_steps_multiplier", 1)
        self.update_val_loss_freq = kwargs.get("update_val_loss_freq", 1)
        self.start_intensive_val_at = kwargs.get("start_intensive_val_at", 0.0)
        if isinstance(self.start_intensive_val_at, float):
            self.start_intensive_val_at = np.clip(self.start_intensive_val_at, 0.0, 1.0)
        elif not isinstance(self.start_intensive_val_at, int):
            raise ValueError("start_intensive_val_at must be a float or an integer.")
        self.hh_memory_size_strategy = kwargs.get("hh_memory_size_strategy", "out_memory_size")
        self.val_dataloader = None
        self.kwargs = kwargs

    def start(self, trainer, **kwargs):
        self.val_dataloader = trainer.state.objects.get("val_dataloader", None)

    def get_hh_memory_size_from_y_batch(self, y_batch) -> int:
        if self.hh_memory_size_strategy == "out_memory_size":
            return y_batch.shape[-2] * self.time_steps_multiplier
        elif self.hh_memory_size_strategy == "foresight_time_steps":
            return y_batch.shape[-2]
        elif isinstance(self.hh_memory_size_strategy, numbers.Number):
            return int(self.hh_memory_size_strategy)
        else:
            raise ValueError(f"Invalid hh_memory_size_strategy: {self.hh_memory_size_strategy}")

    def on_batch_begin(self, trainer, **kwargs):
        self._hh_memory_size_cache = trainer.model.hh_memory_size
        self._out_memory_size_cache = trainer.model.out_memory_size
        self._foresight_time_steps_cache = trainer.model.foresight_time_steps
        if not trainer.model.training:
            n_times_steps = trainer.state.y_batch.shape[-2]
            trainer.model.hh_memory_size = self.get_hh_memory_size_from_y_batch(trainer.state.y_batch)
            trainer.model.out_memory_size = int(n_times_steps * self.time_steps_multiplier)
            trainer.model.foresight_time_steps = trainer.model.out_memory_size

    def on_batch_end(self, trainer, **kwargs):
        trainer.model.hh_memory_size = self._hh_memory_size_cache
        trainer.model.out_memory_size = self._out_memory_size_cache
        trainer.model.foresight_time_steps = self._foresight_time_steps_cache

    def on_train_end(self, trainer, **kwargs):
        if isinstance(self.start_intensive_val_at, float):
            at = int(self.start_intensive_val_at * trainer.state.n_iterations)
        else:
            at = self.start_intensive_val_at
        if (trainer.state.iteration >= at) or ((trainer.state.iteration + 1) % self.update_val_loss_freq == 0):
            trainer.state.objects["val_dataloader"] = self.val_dataloader
        else:
            trainer.state.objects["val_dataloader"] = None
	


