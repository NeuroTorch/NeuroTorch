import warnings
from collections import OrderedDict
from copy import deepcopy, copy
from typing import Iterable, Optional, List, Callable, Dict, Any, Union, NamedTuple, Generator

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..transforms.base import to_numpy, ToTensor
from ..learning_algorithms.bptt import BPTT
from ..callbacks import CheckpointManager, LoadCheckpointMode, TrainingHistory
from ..callbacks.base_callback import BaseCallback, CallbacksList
from ..learning_algorithms.learning_algorithm import LearningAlgorithm
from ..modules import BaseModel
from ..regularization import BaseRegularization, RegularizationList
from ..utils.collections import unpack_x_hh_y


class CurrentTrainingState(NamedTuple):
    r"""
    This class is used to store the current training state. It is extremely useful for the callbacks
    to access the current training state and to personalize the training process.

    :Attributes:
        - **n_iterations** (int): The total number of iterations.
        - **iteration** (int): The current iteration.
        - **epoch** (int): The current epoch.
        - **batch** (int): The current batch.
        - **x_batch** (Any): The current input batch.
        - **hh_batch** (Any): The current hidden state batch.
        - **y_batch** (Any): The current target batch.
        - **pred_batch** (Any): The current prediction.
        - **batch_loss** (float): The current loss.
        - **batch_is_train** (bool): Whether the current batch is a training batch.
        - **train_loss** (float): The current training loss.
        - **val_loss** (float): The current validation loss.
        - **itr_metrics** (Dict[str, Any]): The current iteration metrics.
        - **stop_training_flag** (bool): Whether the training should be stopped.
        - **info** (Dict[str, Any]): Any additional information. This is useful to communicate between callbacks.
        - **objects** (Dict[str, Any]): Any additional objects. This is useful to manage objects between callbacks.
            Note: In general, the train_dataloader and val_dataloader should be stored here.

    """
    n_iterations: Optional[int] = None
    iteration: Optional[int] = None
    n_epochs: Optional[int] = None
    epoch: Optional[int] = None
    epoch_loss: Optional[Any] = None
    batch: Optional[int] = None
    x_batch: Optional[Any] = None
    hh_batch: Optional[Any] = None
    y_batch: Optional[Any] = None
    pred_batch: Optional[Any] = None
    batch_loss: Optional[Any] = None
    batch_is_train: Optional[bool] = None
    train_loss: Optional[Any] = None
    val_loss: Optional[Any] = None
    train_metrics: Optional[Any] = None
    val_metrics: Optional[Any] = None
    itr_metrics: Optional[Dict[str, Any]] = {}
    stop_training_flag: bool = False
    info: Dict[str, Any] = {}
    objects: Dict[str, Any] = {}

    def __getstate__(self):
        not_picklable = ["info", "objects"]
        d = {k: v for k, v in self._asdict().items() if k not in not_picklable}
        return d

    @staticmethod
    def get_null_state() -> "CurrentTrainingState":
        return CurrentTrainingState()

    def update(self, **kwargs) -> "CurrentTrainingState":
        self_dict = self._asdict()
        assert all(k in self_dict for k in kwargs)
        self_dict.update(kwargs)
        return CurrentTrainingState(**self_dict)


TrainingState = CurrentTrainingState  # Alias


class Trainer:
    """
    Trainer class. This class is used to train a model.

    TODO: Add the possibility to pass a callable as the `predict_method`.
    """
    def __init__(
            self,
            model: torch.nn.Module,
            *,
            predict_method: str = "__call__",
            criterion: Optional[Union[Dict[str, Union[torch.nn.Module, Callable]], torch.nn.Module, Callable]] = None,
            regularization: Optional[Union[BaseRegularization, RegularizationList, Iterable[BaseRegularization]]] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            learning_algorithm: Optional[LearningAlgorithm] = None,
            regularization_optimizer: Optional[torch.optim.Optimizer] = None,
            metrics: Optional[List[Callable]] = None,
            callbacks: Optional[Union[List[BaseCallback], CallbacksList, BaseCallback]] = None,
            device: Optional[torch.device] = None,
            verbose: bool = True,
            **kwargs
    ):
        """
        Constructor for Trainer.

        :param model: Model to train.
        :param criterion: Loss function(s) to use. Deprecated, use `learning_algorithm` instead.
        :param regularization: Regularization(s) to use. In NeuroTorch, there are two ways to do regularization:
            1. Regularization can be specified in the layers with the 'update_regularization_loss' method. This
            regularization will be performed by the same optimizer as the main loss. This way is useful when you
            want a regularization that depends on the model output or hidden state.
            2. Regularization can be specified in the trainer with the 'regularization' parameter. This regularization
            will be performed by a separate optimizer named 'regularization_optimizer'. This way is useful when you
            want a regularization that depends only on the model parameters and when you want to control the
            learning rate of the regularization independently of the main loss.

            Note: This parameter will be deprecated and remove in a future version. The regularization will be
                specified in the learning algorithm and/or in the callbacks.

        :param optimizer: Optimizer to use for the main loss. Deprecated. Use learning_algorithm instead.
        :param learning_algorithm: Learning algorithm to use for the main loss. This learning algorithm can be given
            in the callbacks list as well. If specified, this learning algorithm will be added to the callbacks list.
            In this case, make sure that the learning algorithm is not added twice. Note that multiple learning
            algorithms can be used in the callbacks list.
        :param regularization_optimizer: Optimizer to use for the regularization loss.
        :param metrics: Metrics to compute during training.
        :param callbacks: Callbacks to use during training. Each callback will be called at different moments,
            see the documentation of :class:`BaseCallback` for more information.
        :param device: Device to use for the training. Default is the device of the model.
        :param verbose: Whether to print information during training.
        :param kwargs: Additional arguments of the training.

        :keyword int n_epochs: The number of epochs to train at each iteration. Default is 1.
        :keyword float lr: Learning rate of the main optimizer. Default is 1e-3.
        :keyword float reg_lr: Learning rate of the regularization optimizer. Default is 1e-2.
        :keyword float weight_decay: Weight decay of the main optimizer. Default is 0.0.
        :keyword bool exec_metrics_on_train: Whether to compute metrics on the train dataset. This is useful when
            you want to save time by not computing the metrics on the train dataset. Default is True.
        :keyword x_transform: Transform to apply to the input data before passing it to the model.
        :keyword y_transform: Transform to apply to the target data before passing it to the model. For example,
            this can be used to convert the target data to a one-hot encoding or to long tensor
            using `nt.ToTensor(dtype=torch.long)`.
        """
        # assert model.is_built, "Model must be built before training"
        self.kwargs = self._set_default_kwargs(kwargs)
        self.model = model
        self.predict_method = predict_method
        assert hasattr(model, predict_method), f"Model {model.__class__} does not have a method named '{predict_method}'"
        assert callable(getattr(model, predict_method)), f"Model method '{model.__class__}.{predict_method}' is not callable"
        self.criterion = self._set_default_criterion(criterion)
        self.regularization = self._set_default_regularization(regularization)
        # self._maybe_add_regularization(self.regularization)
        self.optimizer = self._set_default_optimizer(optimizer)
        self.regularization_optimizer = self._set_default_reg_optimizer(regularization_optimizer)
        self.metrics = self._set_default_metrics(metrics)
        self.callbacks: CallbacksList = self._set_default_callbacks(callbacks)
        self._maybe_add_learning_algorithm(learning_algorithm)
        self.sort_callbacks_()
        self.device = self._set_default_device(device)
        self.verbose = verbose
        self.training_history: TrainingHistory = self.training_histories[0]
        self.current_training_state = CurrentTrainingState()

        self.x_transform = kwargs.get("x_transform", ToTensor())
        self.y_transform = kwargs.get("y_transform", ToTensor())

        self._load_checkpoint_mode = None
        self._force_overwrite = None

    @property
    def network(self):
        """
        Alias for the model.

        :return: The :attr:`model` attribute.
        """
        return self.model

    @network.setter
    def network(self, value):
        """
        Alias for the model.

        :param value: The new value for the :attr:`model` attribute.
        :return: None
        """
        self.model = value

    @property
    def state(self):
        """
        Alias for the :attr:`current_training_state` attribute.

        :return: The :attr:`current_training_state`
        """
        return self.current_training_state

    @property
    def load_checkpoint_mode(self):
        return self._load_checkpoint_mode

    @load_checkpoint_mode.setter
    def load_checkpoint_mode(self, value: LoadCheckpointMode):
        self._load_checkpoint_mode = value

    @property
    def force_overwrite(self):
        return self._force_overwrite

    @property
    def training_histories(self) -> CallbacksList:
        return CallbacksList(list(filter(lambda x: isinstance(x, TrainingHistory), self.callbacks)))

    @property
    def checkpoint_managers(self) -> CallbacksList:
        return CallbacksList(list(filter(lambda x: isinstance(x, CheckpointManager), self.callbacks)))

    @property
    def learning_algorithms(self) -> CallbacksList:
        return CallbacksList(list(filter(lambda x: isinstance(x, LearningAlgorithm), self.callbacks)))

    @staticmethod
    def _set_default_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs.setdefault("n_epochs", 1)
        kwargs.setdefault("lr", 1e-3)
        kwargs.setdefault("reg_lr", 1e-2)
        kwargs.setdefault("weight_decay", 0.0)
        kwargs.setdefault("batch_size", 256)
        kwargs.setdefault("exec_metrics_on_train", True)

        assert kwargs["batch_size"] > 0, "batch_size must be positive"
        return kwargs

    def _set_default_optimizer(self, optimizer: Optional[torch.optim.Optimizer]) -> torch.optim.Optimizer:
        warnings.warn("The 'optimizer' parameter is deprecated. Use the 'callbacks' parameter instead.", DeprecationWarning)
        # if optimizer is None:
        # 	optimizer = torch.optim.Adam(
        # 		self.model.parameters(),
        # 		lr=self.kwargs["lr"],
        # 		weight_decay=self.kwargs["weight_decay"],
        # 	)
        return optimizer

    def _maybe_add_learning_algorithm(self, learning_algorithm: Optional[LearningAlgorithm]) -> None:
        if len(self.learning_algorithms) == 0 and learning_algorithm is None:
            learning_algorithm = BPTT(optimizer=self.optimizer, criterion=self.criterion)
        if learning_algorithm is not None:
            self.callbacks.append(learning_algorithm)

    def _maybe_add_regularization(self, regularization: Optional[RegularizationList]) -> None:
        if regularization is not None:
            self.callbacks.append(regularization)

    def _set_default_reg_optimizer(self, optimizer: Optional[torch.optim.Optimizer]) -> torch.optim.Optimizer:
        warnings.warn("The 'regularization_optimizer' parameter is deprecated. Use the 'callbacks' parameter instead.", DeprecationWarning)
        if optimizer is None and self.regularization is not None:
            optimizer = torch.optim.SGD(
                self.regularization.parameters(),
                lr=self.kwargs["reg_lr"],
                weight_decay=0.0,
            )
        return optimizer

    def _set_default_metrics(self, metrics: Optional[List[Callable]]):
        if metrics is None:
            metrics = []
        return metrics

    def _set_default_criterion(self, criterion: Optional[torch.nn.Module]) -> torch.nn.Module:
        warnings.warn("The 'criterion' parameter is deprecated. Use the 'callbacks' parameter instead.", DeprecationWarning)
        # if criterion is None:
        # 	if isinstance(self.model.output_sizes, dict):
        # 		criterion = {
        # 			k: torch.nn.MSELoss() for k in self.model.output_sizes
        # 		}
        # 	elif isinstance(self.model.output_sizes, int):
        # 		criterion = torch.nn.MSELoss()
        # 	else:
        # 		raise ValueError("Unknown criterion type")
        return criterion

    def _set_default_regularization(
            self,
            regularization: Optional[Union[BaseRegularization, RegularizationList, Iterable[BaseRegularization]]]
    ) -> Optional[RegularizationList]:
        warnings.warn("The 'regularization' parameter is deprecated. Use the 'callbacks' parameter instead.", DeprecationWarning)
        if regularization is None:
            pass
        elif isinstance(regularization, BaseRegularization):
            regularization = RegularizationList([regularization])
        elif isinstance(regularization, RegularizationList):
            pass
        elif isinstance(regularization, Iterable):
            regularization = RegularizationList(regularization)
        return regularization

    def _set_default_device(self, device: Optional[torch.device]) -> torch.device:
        if device is None:
            if hasattr(self.model, "device"):
                device = self.model.device
            else:
                device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        return device

    @staticmethod
    def _set_default_callbacks(
            callbacks: Optional[Union[List[BaseCallback], CallbacksList, BaseCallback]]
    ) -> CallbacksList:
        if callbacks is None:
            callbacks = []
        if isinstance(callbacks, BaseCallback):
            callbacks = [callbacks]
        if not any([isinstance(callback, TrainingHistory) for callback in callbacks]):
            callbacks.append(TrainingHistory())
        return CallbacksList(callbacks)

    def update_state_(self, **kwargs):
        self.current_training_state = self.current_training_state.update(**kwargs)

    def update_objects_state_(self, **kwargs):
        self.update_state_(objects={**self.current_training_state.objects, **kwargs})

    def update_info_state_(self, **kwargs):
        self.update_state_(info={**self.current_training_state.info, **kwargs})

    def update_itr_metrics_state_(self, **kwargs):
        self.update_state_(itr_metrics={**self.current_training_state.itr_metrics, **kwargs})

    def sort_callbacks_(self, reverse: bool = False) -> CallbacksList:
        """
        Sort the callbacks by their priority. The higher the priority, the earlier the callback is called. In general,
        the callbacks will be sorted in the following order:
            1. TrainingHistory callbacks;
            2. Others callbacks;
            3. CheckpointManager callbacks.

        :param reverse: Whether to reverse the order of the callbacks. Default is False.
        :type reverse: bool
        :return: The sorted callbacks.
        :rtype: CallbacksList
        """
        self.callbacks.sort_callbacks_(reverse=reverse)
        return self.callbacks

    def load_state(self):
        """
        Load the state of the trainer from the checkpoint.
        """
        if self.checkpoint_managers:
            main_checkpoint_manager: CheckpointManager = self.checkpoint_managers[0]
            checkpoint = main_checkpoint_manager.curr_checkpoint
            if checkpoint:
                self.callbacks.load_checkpoint_state(self, checkpoint)

    def train(
            self,
            train_dataloader: DataLoader,
            val_dataloader: Optional[DataLoader] = None,
            n_iterations: Optional[int] = None,
            *,
            n_epochs: int = 1,
            load_checkpoint_mode: LoadCheckpointMode = None,
            force_overwrite: bool = False,
            p_bar_position: Optional[int] = None,
            p_bar_leave: Optional[bool] = None,
            **kwargs
    ) -> TrainingHistory:
        """
        Train the model.

        :param train_dataloader: The dataloader for the training set. It contains the training data.
        :type train_dataloader: DataLoader
        :param val_dataloader: The dataloader for the validation set. It contains the validation data.
        :type val_dataloader: Optional[DataLoader]
        :param n_iterations: The number of iterations to train the model. An iteration is a pass over the training set
            and the validation set. If None, the model will be trained until the training is stopped by the user.
        :type n_iterations: Optional[int]
        :param n_epochs: The number of epochs to train the model. An epoch is a pass over the training set. The
            nomenclature here is different from what is usually used elsewhere. Here, an epoch is a pass over the
            training set, while an iteration is a pass over the training set and the validation set. In other words,
            if `n_iterations=1` and `n_epochs=10`, the trainer will pass 10 times over the training set  and 1 time
            over the validation set (this will constitute 1 iteration). If `n_iterations=10` and `n_epochs=1`, the
            trainer will pass 10 times over the training set and 10 times over the validation set (this will constitute
            10 iterations). The nuance between those terms is really important when is comes to reinforcement learning.
            Default is 1.
        :type n_epochs: int
        :param load_checkpoint_mode: The mode to use when loading the checkpoint.
        :type load_checkpoint_mode: LoadCheckpointMode
        :param force_overwrite: Whether to force overwriting the checkpoint. Be careful when using this option, as it
            will destroy the previous checkpoint folder. Default is False.
        :type force_overwrite: bool
        :param p_bar_position: The position of the progress bar. See tqdm documentation for more information.
        :type p_bar_position: Optional[int]
        :param p_bar_leave: Whether to leave the progress bar. See tqdm documentation for more information.
        :type p_bar_leave: Optional[bool]
        :param kwargs: Additional keyword arguments.

        :return: The training history.
        """
        self._load_checkpoint_mode = load_checkpoint_mode
        self._force_overwrite = force_overwrite
        self.kwargs.update(kwargs)
        self.update_state_(
            n_iterations=n_iterations,
            n_epochs=n_epochs,
            objects={
                **self.current_training_state.objects,
                **{"train_dataloader": train_dataloader, "val_dataloader": val_dataloader}
            }
        )
        self.sort_callbacks_()
        self.callbacks.start(self)
        self.load_state()
        if self.current_training_state.iteration is None:
            self.update_state_(iteration=0)
        if len(self.training_history) > 0:
            self.update_itr_metrics_state_(**self.training_history.get_item_at(-1))
        else:
            self.update_state_(itr_metrics={})
        p_bar = tqdm(
            initial=self.current_training_state.iteration,
            total=self.current_training_state.n_iterations,
            desc=kwargs.get("desc", "Training"),
            disable=not self.verbose,
            position=p_bar_position,
            unit="itr",
            leave=p_bar_leave
        )
        self.update_objects_state_(p_bar=p_bar)
        for i in self._iterations_generator(p_bar):
            self.update_state_(iteration=i)
            self.callbacks.on_iteration_begin(self)
            train_dataloader = self.current_training_state.objects["train_dataloader"]
            val_dataloader = self.current_training_state.objects["val_dataloader"]
            itr_loss = self._exec_iteration(train_dataloader, val_dataloader)
            if self.kwargs["exec_metrics_on_train"]:
                itr_train_metrics = self._exec_metrics(train_dataloader, prefix="train")
            else:
                itr_train_metrics = {}
            if val_dataloader is not None:
                itr_val_metrics = self._exec_metrics(val_dataloader, prefix="val")
            else:
                itr_val_metrics = {}
            self.update_itr_metrics_state_(**dict(**itr_loss, **itr_train_metrics, **itr_val_metrics))
            postfix = {f"{k}": f"{v:.5e}" for k, v in self.state.itr_metrics.items()}
            postfix.update(self.callbacks.on_pbar_update(self))
            self.callbacks.on_iteration_end(self)
            p_bar.set_postfix(postfix)
            if self.current_training_state.stop_training_flag:
                p_bar.set_postfix(OrderedDict(**{"stop_flag": "True"}, **postfix))
                break
        self.callbacks.close(self)
        p_bar.close()
        return self.training_history

    def _iterations_generator(self, p_bar: tqdm) -> Generator:
        """
        Generator that yields the current iteration and updates the state and the p_bar.

        :return: The current iteration.
        """
        while self.current_training_state.iteration < self._get_numeric_n_iterations():
            yield self.current_training_state.iteration
            self.update_state_(iteration=self.current_training_state.iteration + 1)
            p_bar.total = self.current_training_state.n_iterations
            p_bar.update()

    def _get_numeric_n_iterations(self) -> int:
        """
        Returns the number of iterations.

        :return: The number of iterations.
        """
        n_iterations = self.current_training_state.n_iterations
        if n_iterations is None:
            n_iterations = np.inf
        return n_iterations

    def _exec_iteration(
            self,
            train_dataloader: DataLoader,
            val_dataloader: Optional[DataLoader] = None
    ) -> Dict[str, float]:
        with torch.no_grad():
            torch.cuda.empty_cache()
        losses = {}

        self.model.train()
        self.callbacks.on_train_begin(self)
        self.update_state_(batch_is_train=True)
        train_losses = []
        for epoch_idx in range(self.current_training_state.n_epochs):
            self.update_state_(epoch=epoch_idx)
            train_losses.append(self._exec_epoch(train_dataloader))
        train_loss = np.mean(train_losses)
        self.update_state_(train_loss=train_loss)
        self.callbacks.on_train_end(self)
        losses["train_loss"] = train_loss

        if val_dataloader is not None:
            with torch.no_grad():
                self.model.eval()
                self.callbacks.on_validation_begin(self)
                self.update_state_(batch_is_train=False)
                val_loss = self._exec_epoch(val_dataloader)
                self.update_state_(val_loss=val_loss)
                self.callbacks.on_validation_end(self)
                losses["val_loss"] = val_loss

        with torch.no_grad():
            torch.cuda.empty_cache()
        return losses

    def _exec_metrics(self, dataloader: torch.utils.data.DataLoader, prefix: str) -> Dict:
        metrics_dict = {}
        for metric in self.metrics:
            m_out = metric(dataloader)
            if isinstance(m_out, dict):
                metrics_dict.update({f"{prefix}_{k}": v for k, v in m_out.items()})
            else:
                metric_name = str(metric)
                if hasattr(metric, "name"):
                    metric_name = metric.name
                elif hasattr(metric, "__name__"):
                    metric_name = metric.__name__
                metrics_dict[f"{prefix}_{metric_name}"] = m_out
        return metrics_dict

    def _exec_epoch(
            self,
            dataloader: DataLoader,
    ) -> float:
        self.callbacks.on_epoch_begin(self)
        batch_losses = []
        for i, entries in enumerate(dataloader):
            x_batch, hh_batch, y_batch = unpack_x_hh_y(entries)
            self.update_state_(batch=i)
            batch_losses.append(to_numpy(self._exec_batch(x_batch, hh_batch, y_batch)))
        mean_loss = np.mean(batch_losses)
        self.callbacks.on_epoch_end(self)
        return mean_loss

    def _exec_batch(
            self,
            x_batch,
            hh_batch,
            y_batch,
    ):
        x_batch = self.x_transform(self._batch_to_dense(self._batch_to_device(x_batch)))
        if hh_batch is not None:
            hh_batch = self.x_transform(self._batch_to_dense(self._batch_to_device(hh_batch)))
        y_batch = self.y_transform(self._batch_to_dense(self._batch_to_device(y_batch)))
        self.update_state_(x_batch=x_batch, hh_batch=hh_batch, y_batch=y_batch)
        self.callbacks.on_batch_begin(self)
        pred_batch = self.get_pred_batch(x_batch, hh_batch)
        self.update_state_(pred_batch=pred_batch)
        if self.model.training:
            self.callbacks.on_optimization_begin(self, x=x_batch, hh_batch=hh_batch, y=y_batch, pred=pred_batch)
            self.callbacks.on_optimization_end(self)
        else:
            self.callbacks.on_validation_batch_begin(self, x=x_batch, hh_batch=hh_batch, y=y_batch, pred=pred_batch)
            self.callbacks.on_validation_batch_end(self)
        self.callbacks.on_batch_end(self)
        batch_loss = self.current_training_state.batch_loss
        if batch_loss is None:
            batch_loss = 0.0
        else:
            batch_loss = batch_loss.item()
        return batch_loss

    def get_pred_batch(
            self,
            x_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
            hh_batch: Optional[Any] = None,
            *args,
            **kwargs
    ):
        """
        Get the prediction of the model on the given batch.

        :param x_batch: The input batch.
        :type x_batch: Union[torch.Tensor, Dict[str, torch.Tensor]]
        :param hh_batch: The hidden state batch.
        :type hh_batch: Optional[Any]
        :param args: Additional arguments to pass to the model.
        :param kwargs: Additional keyword arguments to pass to the model.
        """
        if hh_batch is not None:
            args = (hh_batch, *args)
        if self.model.training:
            out = getattr(self.model, self.predict_method)(x_batch, *args, **kwargs)
        else:
            with torch.no_grad():
                out = getattr(self.model, self.predict_method)(x_batch, *args, **kwargs)
        return out

    def apply_criterion_on_batch(
            self,
            x_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
            y_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
            pred_batch: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        if isinstance(pred_batch, (tuple, list)):
            pred = pred_batch[0]
        elif isinstance(pred_batch, torch.Tensor):
            pred = pred_batch
        else:
            raise ValueError(f"Unsupported output type: {type(pred_batch)}")

        if isinstance(self.criterion, dict):
            if isinstance(y_batch, torch.Tensor):
                y_batch = {k: y_batch for k in self.criterion}
            if isinstance(pred, torch.Tensor):
                pred = {k: pred for k in self.criterion}
            assert isinstance(pred, dict) and isinstance(y_batch, dict) and isinstance(pred, dict), \
                "If criterion is a dict, pred, y_batch and pred must be a dict too."
            batch_loss = sum([
                self.criterion[k](pred[k], y_batch[k].to(self.device))
                for k in self.criterion
            ])
        else:
            if isinstance(pred, dict) and len(pred) == 1:
                pred = pred[list(pred.keys())[0]]
            batch_loss = self.criterion(pred, y_batch.to(self.device))
        return batch_loss

    def _batch_to_dense(self, batch):
        if isinstance(batch, dict):
            return {k: self._batch_to_dense(v) for k, v in batch.items()}
        if isinstance(batch, torch.Tensor) and batch.is_sparse:
            return batch.to_dense()
        return batch

    def _batch_to_device(self, batch):
        if isinstance(batch, dict):
            return {k: self._batch_to_device(v) for k, v in batch.items()}
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        return batch

    def __repr__(self):
        repr_ = f"{self.__class__.__name__}("
        repr_ += f"\n\tmodel={self.model}, "
        repr_ += f"\n\tpredict_method={self.predict_method}, "
        repr_ += f"\n\tcallbacks={self.callbacks}"
        repr_ += f")@{self.device}"
        return repr_



