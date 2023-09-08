from typing import Iterable, Optional, Iterator, Dict, Any, List


class BaseCallback:
    """
    Class used to create a callback that can be used to monitor or modify the training process.

    Training Phases:
        - Iteration: One full pass through the training dataset and the validation dataset.
        - Epoch: One full pass through the training dataset or the validation dataset.
        - Batch: One forward pass through the network.
        - Train: One full pass through the training dataset.
        - Validation: One full pass through the validation dataset.

    Callbacks methods are called in the following order:
        - :meth:`start`
        - :meth:`load_checkpoint_state`
        * Executes n_iterations times:
            - :meth:`on_iteration_begin`
            - :meth:`on_train_begin`
            * Executes n_epochs times:
                - :meth:`on_epoch_begin`
                * Executes n_batches times:
                    - :meth:`on_batch_begin`
                    - :meth:`on_optimization_begin`
                    - :meth:`on_optimization_end`
                    - :meth:`on_batch_end`
                - :meth:`on_epoch_end`
            - :meth:`on_train_end`
            - :meth:`on_validation_begin`
            - :meth:`on_epoch_begin`
            * Executes n_batches times:
                - :meth:`on_batch_begin`
                - :meth:`on_validation_batch_begin`
                - :meth:`on_validation_batch_end`
                - :meth:`on_batch_end`
            - :meth:`on_epoch_end`
            - :meth:`on_validation_end`
            - :meth:`on_iteration_end`
        - :meth:`close`

    :Note: The special method :meth:`get_checkpoint_state` is called by the object :class:`CheckpointManager` to
        save the state of the callback in the checkpoint file. The when this method is called is then determined by
        the :class:`CheckpointManager` object if it is used in the trainer callbacks. In the same way, the method
        :meth:`load_checkpoint_state` is called by the :class:`CheckpointManager` to load the state of the callback
        from the checkpoint file if it is used in the trainer callbacks.

    :Note: The special method :meth:`__del__` is called when the callback is deleted. This is used to call the
        :meth:`close` method if it was not called before.

    :Attributes:
        - :attr:`Priority`: The priority of the callback. The lower the priority, the earlier the callback is called.
            Default is 10.
    """
    DEFAULT_PRIORITY = 10
    DEFAULT_LOW_PRIORITY = 100
    DEFAULT_MEDIUM_PRIORITY = 50
    DEFAULT_HIGH_PRIORITY = 0

    instance_counter = 0

    UNPICKEABLE_ATTRIBUTES = ["trainer"]

    def __init__(
            self,
            priority: Optional[int] = None,
            name: Optional[str] = None,
            save_state: bool = True,
            load_state: Optional[bool] = None,
            **kwargs
    ):
        """
        :param priority: The priority of the callback. The lower the priority, the earlier the callback is called.
            At the beginning of the training the priorities of the callbacks are reversed for the :meth:`load_state`
            method. Default is 10.
        :type priority: int, optional
        :param name: The name of the callback. If None, the name is set to the class name. Default is None.
        :type name: str, optional
        :param save_state: If True, the state of the callback is saved in the checkpoint file. Default is True.
        :type save_state: bool, optional
        :param load_state: If True, the state of the callback is loaded from the checkpoint file. Default is equal to
            save_state.
        :type load_state: bool, optional
        """
        self.kwargs = kwargs
        self.priority = priority if priority is not None else self.DEFAULT_PRIORITY
        self.instance_id = self.instance_counter
        self.name = name if name is not None else f"{self.__class__.__name__}<{self.instance_id}>"
        self.save_state = save_state
        self.load_state = load_state if load_state is not None else save_state
        self.__class__.instance_counter += 1
        self.trainer = None
        self._start_flag = False
        self._close_flag = False

    def load_checkpoint_state(self, trainer, checkpoint: dict, **kwargs):
        """
        Loads the state of the callback from a dictionary.

        :param trainer: The trainer.
        :type trainer: Trainer
        :param checkpoint: The dictionary containing all the states of the trainer.
        :type checkpoint: dict

        :return: None
        """
        if self.load_state and checkpoint is not None:
            state = checkpoint.get(self.name, None)
            if state is not None:
                self.__dict__.update(state)

    def get_checkpoint_state(self, trainer, **kwargs) -> object:
        """
        Get the state of the callback. This is called when the checkpoint manager saves the state of the trainer.
        Then this state is saved in the checkpoint file with the name of the callback as the key.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: The state of the callback.
        :rtype: An pickleable object.
        """
        if self.save_state:
            return {k: v for k, v in self.__dict__.items() if k not in self.UNPICKEABLE_ATTRIBUTES}

    def start(self, trainer, **kwargs):
        """
        Called when the training starts. This is the first callback called.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        self.trainer = trainer
        self._start_flag = True

    def close(self, trainer, **kwargs):
        """
        Called when the training ends. This is the last callback called.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        self._close_flag = True

    def on_train_begin(self, trainer, **kwargs):
        """
        Called when the train phase of an iteration starts. The train phase is defined as a full pass through the
        training dataset.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        pass

    def on_train_end(self, trainer, **kwargs):
        """
        Called when the train phase of an iteration ends. The train phase is defined as a full pass through the
        training dataset.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        pass

    def on_epoch_begin(self, trainer, **kwargs):
        """
        Called when an epoch starts. An epoch is defined as one full pass through the training dataset or
        the validation dataset.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        pass

    def on_epoch_end(self, trainer, **kwargs):
        """
        Called when an epoch ends. An epoch is defined as one full pass through the training dataset or
        the validation dataset.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        pass

    def on_batch_begin(self, trainer, **kwargs):
        """
        Called when a batch starts. The batch is defined as one forward pass through the network.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        pass

    def on_batch_end(self, trainer, **kwargs):
        """
        Called when a batch ends. The batch is defined as one forward pass through the network.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        pass

    def on_validation_begin(self, trainer, **kwargs):
        """
        Called when the validation phase of an iteration starts. The validation phase is defined as a full pass through
        the validation dataset.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return:
        """
        pass

    def on_validation_end(self, trainer, **kwargs):
        """
        Called when the validation phase of an iteration ends. The validation phase is defined as a full pass through
        the validation dataset.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        pass

    def on_iteration_begin(self, trainer, **kwargs):
        """
        Called when an iteration starts. An iteration is defined as one full pass through the training dataset and
        the validation dataset.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        pass

    def on_iteration_end(self, trainer, **kwargs):
        """
        Called when an iteration ends. An iteration is defined as one full pass through the training dataset and
        the validation dataset.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        pass

    def on_optimization_begin(self, trainer, **kwargs):
        """
        Called when the optimization phase of an iteration starts. The optimization phase is defined as
        the moment where the model weights are updated.

        :param trainer: The trainer.
        :type trainer: Trainer
        :param kwargs: Additional arguments.

        :keyword x: The input data.
        :keyword y: The target data.
        :keyword pred: The predicted data.

        :return: None
        """
        pass

    def on_optimization_end(self, trainer, **kwargs):
        """
        Called when the optimization phase of an iteration ends. The optimization phase is defined as
        the moment where the model weights are updated.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        pass

    def on_validation_batch_begin(self, trainer, **kwargs):
        """
        Called when the validation batch starts. The validation batch is defined as one forward pass through the network
        on the validation dataset. This is used to update the batch loss and metrics on the validation dataset.

        :param trainer: The trainer.
        :type trainer: Trainer
        :param kwargs: Additional arguments.

        :keyword x: The input data.
        :keyword y: The target data.
        :keyword pred: The predicted data.

        :return: None
        """
        pass

    def on_validation_batch_end(self, trainer, **kwargs):
        """
        Called when the validation batch ends. The validation batch is defined as one forward pass through the network
        on the validation dataset. This is used to update the batch loss and metrics on the validation dataset.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        pass

    def on_trajectory_end(self, trainer, trajectory, **kwargs) -> List[Dict[str, Any]]:
        """
        Called when a trajectory ends. This is used in reinforcement learning to update the trajectory loss and metrics.
        Must return a list of dictionaries containing the trajectory metrics. The list must have the same length as the
        trajectory. Each item in the list will update the attribute `others` of the corresponding Experience.

        :param trainer: The trainer.
        :type trainer: Trainer
        :param trajectory: The trajectory i.e. the sequence of Experiences.
        :type trajectory: Trajectory
        :param kwargs: Additional arguments.

        :return: A list of dictionaries containing the trajectory metrics.
        """
        pass

    def on_pbar_update(self, trainer, **kwargs) -> dict:
        """
        Called when the progress bar is updated.

        :param trainer: The trainer.
        :type trainer: Trainer
        :param kwargs: Additional arguments.

        :return: None
        """
        return {}

    def __del__(self):
        try:
            if (not self._close_flag) and self.trainer is not None:
                self.close(self.trainer)
        except:
            pass
        self.__class__.instance_counter -= 1

    def extra_repr(self) -> str:
        return ""

    def __repr__(self):
        repr_str = f"{self.name}: ("
        repr_str += f"priority={self.priority}, "
        repr_str += f"save_state={self.save_state}, "
        repr_str += f"load_state={self.load_state}, "
        repr_str += self.extra_repr()
        repr_str += f")"
        return repr_str


class CallbacksList:
    """
    This class is used to store the callbacks that are used during the training. Each callback of the list is called
    in the order they are stored in the list.

    :Attributes:
        - :attr:`callbacks` (List[BaseCallback]): The callbacks to use.
    """
    def __init__(self, callbacks: Optional[Iterable[BaseCallback]] = None):
        """
        Constructor of the CallbacksList class.

        :param callbacks: The callbacks to use.
        :type callbacks: Iterable[BaseCallback]
        """
        if callbacks is None:
            callbacks = []
        assert isinstance(callbacks, Iterable), "callbacks must be an Iterable"
        assert all(isinstance(callback, BaseCallback) for callback in callbacks), \
            "All callbacks must be instances of BaseCallback"
        self.callbacks = list(callbacks)
        self._length = len(self.callbacks)
        self.sort_callbacks_()

    def sort_callbacks_(self, reverse: bool = False) -> 'CallbacksList':
        """
        Sorts the callbacks by their priority.

        :param reverse: If True, the callbacks are sorted in descending order.
        :type reverse: bool

        :return: self
        :rtype: CallbacksList
        """
        self.callbacks.sort(key=lambda callback: callback.priority, reverse=reverse)
        return self

    def __getitem__(self, item: int) -> BaseCallback:
        """
        Get a callback from the list.

        :param item: The index of the callback to get.
        :type item: int

        :return: The callback at the given index.
        :rtype: BaseCallback
        """
        return self.callbacks[item]

    def __iter__(self) -> Iterator[BaseCallback]:
        """
        Get an iterator over the callbacks.

        :return: An iterator over the callbacks.
        :rtype: Iterator[BaseCallback]
        """
        return iter(self.callbacks)

    def __len__(self) -> int:
        """
        Get the number of callbacks in the list.

        :return: The number of callbacks in the list.
        :rtype: int
        """
        return self._length

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}(\n"
        for callback in self.callbacks:
            repr_str += f"\t{repr(callback)}, \n"
        repr_str += f")"
        return repr_str

    def append(self, callback: BaseCallback):
        """
        Append a callback to the list.

        :param callback: The callback to append.
        :type callback: BaseCallback

        :return: None
        """
        assert isinstance(callback, BaseCallback), "callback must be an instance of BaseCallback"
        self.callbacks.append(callback)
        self._length += 1
        self.sort_callbacks_()

    def remove(self, callback: BaseCallback):
        """
        Remove a callback from the list.

        :param callback: The callback to remove.
        :type callback: BaseCallback

        :return: None
        """
        assert isinstance(callback, BaseCallback), "callback must be an instance of BaseCallback"
        self.callbacks.remove(callback)
        self._length -= 1
        self.sort_callbacks_()

    def load_checkpoint_state(self, trainer, checkpoint: dict, **kwargs):
        """
        Loads the state of the callback from a dictionary.

        :param trainer: The trainer.
        :type trainer: Trainer
        :param checkpoint: The dictionary containing all the states of the trainer.
        :type checkpoint: dict

        :return: None
        """
        for callback in self.callbacks:
            callback.load_checkpoint_state(trainer, checkpoint, **kwargs)

    def get_checkpoint_state(self, trainer, **kwargs) -> Dict[str, Any]:
        """
        Collates the states of the callbacks. This is called when the checkpoint manager saves the state of the trainer.
        Then those states are saved in the checkpoint file with the name of the callback as the key.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: The state of the callback.
        :rtype: An pickleable dict.
        """
        states = {
            callback.name: callback.get_checkpoint_state(trainer, **kwargs)
            for callback in self.callbacks
        }
        states = {key: value for key, value in states.items() if value is not None}
        return states

    def start(self, trainer, **kwargs):
        """
        Called when the trainer starts.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        for callback in self.callbacks:
            callback.start(trainer, **kwargs)

    def close(self, trainer, **kwargs):
        """
        Called when the trainer closes.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        for callback in self.callbacks:
            callback.close(trainer, **kwargs)

    def on_train_begin(self, trainer, **kwargs):
        """
        Called when the train phase of an iteration starts. The train phase is defined as a full pass through the
        training dataset.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        for callback in self.callbacks:
            callback.on_train_begin(trainer, **kwargs)

    def on_train_end(self, trainer, **kwargs):
        """
        Called when the train phase of an iteration ends. The train phase is defined as a full pass through the
        training dataset.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        for callback in self.callbacks:
            callback.on_train_end(trainer, **kwargs)

    def on_epoch_begin(self, trainer, **kwargs):
        """
        Called when an epoch starts. An epoch is defined as one full pass through the training dataset or
        the validation dataset.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, **kwargs)

    def on_epoch_end(self, trainer, **kwargs):
        """
        Called when an epoch ends. An epoch is defined as one full pass through the training dataset or
        the validation dataset.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, **kwargs)

    def on_batch_begin(self, trainer, **kwargs):
        """
        Called when a batch starts. The batch is defined as one forward pass through the network.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        for callback in self.callbacks:
            callback.on_batch_begin(trainer, **kwargs)

    def on_batch_end(self, trainer, **kwargs):
        """
        Called when a batch ends. The batch is defined as one forward pass through the network.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        for callback in self.callbacks:
            callback.on_batch_end(trainer, **kwargs)

    def on_validation_begin(self, trainer, **kwargs):
        """
        Called when the validation phase of an iteration starts. The validation phase is defined as a full pass through
        the validation dataset.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        for callback in self.callbacks:
            callback.on_validation_begin(trainer, **kwargs)

    def on_validation_end(self, trainer, **kwargs):
        """
        Called when the validation phase of an iteration ends. The validation phase is defined as a full pass through
        the validation dataset.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        for callback in self.callbacks:
            callback.on_validation_end(trainer, **kwargs)

    def on_iteration_begin(self, trainer, **kwargs):
        """
        Called when an iteration starts. An iteration is defined as one full pass through the training dataset and
        the validation dataset.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        for callback in self.callbacks:
            callback.on_iteration_begin(trainer, **kwargs)

    def on_iteration_end(self, trainer, **kwargs):
        """
        Called when an iteration ends. An iteration is defined as one full pass through the training dataset and
        the validation dataset.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        for callback in self.callbacks:
            callback.on_iteration_end(trainer, **kwargs)

    def on_optimization_begin(self, trainer, **kwargs):
        """
        Called when the optimization phase of an iteration starts. The optimization phase is defined as
        the moment where the model weights are updated.

        :param trainer: The trainer.
        :type trainer: Trainer
        :param kwargs: Additional arguments.

        :return: None
        """
        for callback in self.callbacks:
            callback.on_optimization_begin(trainer, **kwargs)

    def on_optimization_end(self, trainer, **kwargs):
        """
        Called when the optimization phase of an iteration ends. The optimization phase is defined as
        the moment where the model weights are updated.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        for callback in self.callbacks:
            callback.on_optimization_end(trainer, **kwargs)

    def on_validation_batch_begin(self, trainer, **kwargs):
        """
        Called when the validation batch starts. The validation batch is defined as one forward pass through the network
        on the validation dataset. This is used to update the batch loss and metrics on the validation dataset.

        :param trainer: The trainer.
        :type trainer: Trainer
        :param kwargs: Additional arguments.

        :keyword x: The input data.
        :keyword y: The target data.
        :keyword pred: The predicted data.

        :return: None
        """
        for callback in self.callbacks:
            callback.on_validation_batch_begin(trainer, **kwargs)

    def on_validation_batch_end(self, trainer, **kwargs):
        """
        Called when the validation batch ends. The validation batch is defined as one forward pass through the network
        on the validation dataset. This is used to update the batch loss and metrics on the validation dataset.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        for callback in self.callbacks:
            callback.on_validation_batch_end(trainer, **kwargs)

    def on_trajectory_end(self, trainer, trajectory, **kwargs) -> List[Dict[str, Any]]:
        """
        Called when a trajectory ends. This is used in reinforcement learning to update the trajectory loss and metrics.
        Must return a  list of dictionaries containing the trajectory metrics. The list must have the same
        length as the trajectory. Each item in the list will update the attribute `others` of the corresponding
        Experience.

        :Note: If the callbacks return the same keys in the dictionaries, the values will be updated, so the
            last callback will prevail.

        :param trainer: The trainer.
        :type trainer: Trainer
        :param trajectory: The trajectory i.e. the sequence of Experiences.
        :type trajectory: Trajectory
        :param kwargs: Additional arguments.

        :return: A list of dictionaries containing the trajectory metrics.
        """
        # re_list = [{}] * len(trajectory)
        re_list = [{} for _ in range(len(trajectory))]
        for callback in self.callbacks:
            callback_return = callback.on_trajectory_end(trainer, trajectory, **kwargs)
            if callback_return is not None:
                assert len(callback_return) == len(trajectory), \
                    "The callback must return a list of dictionaries with the same length as the trajectory."
                for i, re in enumerate(callback_return):
                    re_list[i].update(re)
        return re_list

    def on_pbar_update(self, trainer, **kwargs) -> dict:
        """
        Called when the progress bar is updated.

        :param trainer: The trainer.
        :type trainer: Trainer
        :param kwargs: Additional arguments.

        :return: None
        """
        re_dict = {}
        for callback in self.callbacks:
            callback_dict = callback.on_pbar_update(trainer, **kwargs)
            if callback_dict is None:
                callback_dict = {}
            elif not isinstance(callback_dict, dict):
                callback_dict = {callback.name: callback_dict}
            re_dict.update(callback_dict)
        return re_dict

