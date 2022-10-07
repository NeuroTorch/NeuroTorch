from collections import OrderedDict
from typing import Iterable, Optional, List, Callable, Dict, Any, Union, NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..callbacks import CheckpointManager, LoadCheckpointMode, TrainingHistory
from ..callbacks.base_callback import BaseCallback, CallbacksList
from ..modules import BaseModel
from ..regularization import BaseRegularization, RegularizationList


class CurrentTrainingState(NamedTuple):
	iteration: Optional[int]
	epoch: Optional[int]
	epoch_loss: Optional[Any]
	batch: Optional[int]
	batch_loss: Optional[Any]
	batch_is_train: Optional[bool]
	train_loss: Optional[Any]
	val_loss: Optional[Any]
	train_metrics: Optional[Any]
	val_metrics: Optional[Any]
	itr_metrics: Optional[Dict[str, Any]]
	stop_training_flag: bool = False
	
	@staticmethod
	def get_null_state() -> "CurrentTrainingState":
		return CurrentTrainingState(
			iteration=None,
			epoch=None,
			epoch_loss=None,
			batch=None,
			batch_loss=None,
			batch_is_train=None,
			train_loss=None,
			val_loss=None,
			train_metrics=None,
			val_metrics=None,
			itr_metrics=None,
			stop_training_flag=False,
		)
	
	def update(
			self,
			*,
			iteration: Optional[int] = None,
			epoch: Optional[int] = None,
			epoch_loss: Optional[Any] = None,
			batch: Optional[int] = None,
			batch_loss: Optional[Any] = None,
			batch_is_train: Optional[bool] = None,
			train_loss: Optional[Any] = None,
			val_loss: Optional[Any] = None,
			train_metrics: Optional[Any] = None,
			val_metrics: Optional[Any] = None,
			itr_metrics: Optional[Dict[str, Any]] = None,
			stop_training_flag: Optional[bool] = None,
	) -> "CurrentTrainingState":
		return CurrentTrainingState(
			iteration=iteration if iteration is not None else self.iteration,
			epoch=epoch if epoch is not None else self.epoch,
			epoch_loss=epoch_loss if epoch_loss is not None else self.epoch_loss,
			batch=batch if batch is not None else self.batch,
			batch_loss=batch_loss if batch_loss is not None else self.batch_loss,
			batch_is_train=batch_is_train if batch_is_train is not None else self.batch_is_train,
			train_loss=train_loss if train_loss is not None else self.train_loss,
			val_loss=val_loss if val_loss is not None else self.val_loss,
			train_metrics=train_metrics if train_metrics is not None else self.train_metrics,
			val_metrics=val_metrics if val_metrics is not None else self.val_metrics,
			itr_metrics=itr_metrics if itr_metrics is not None else self.itr_metrics,
			stop_training_flag=stop_training_flag if stop_training_flag is not None else self.stop_training_flag,
		)


class Trainer:
	def __init__(
			self,
			model: BaseModel,
			criterion: Optional[Union[Dict[str, Union[torch.nn.Module, Callable]], torch.nn.Module, Callable]] = None,
			regularization: Optional[Union[BaseRegularization, RegularizationList, Iterable[BaseRegularization]]] = None,
			optimizer: Optional[torch.optim.Optimizer] = None,
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
		:param criterion: Loss function(s) to use.
		:param regularization: Regularization(s) to use. In NeuroTorch, there are two ways to do regularization:
			1. Regularization can be specified in the layers with the 'update_regularization_loss' method. This
			regularization will be performed by the same optimizer as the main loss. This way is useful when you
			want a regularization that depends on the model output or hidden state.
			2. Regularization can be specified in the trainer with the 'regularization' parameter. This regularization
			will be performed by a separate optimizer named 'regularization_optimizer'. This way is useful when you
			want a regularization that depends only on the model parameters and when you want to control the
			learning rate of the regularization independently of the main loss.
		:param optimizer: Optimizer to use for the main loss.
		:param regularization_optimizer: Optimizer to use for the regularization loss.
		:param metrics: Metrics to compute during training.
		:param callbacks: Callbacks to use during training. Each callback will be called at the following moments:
			1. At the beginning of the train call.
			2. At the beginning of each iteration. An iteration is defined as one full pass through the training
			dataset and the validation dataset.
			3. At the beginning of each epoch. An epoch is defined as one full pass through a dataset (train or valid).
			4. At the beginning of each batch. The batch is defined as one forward pass through the network.
			5. At the end of each batch.
			6. At the end of each epoch.
			7. At the end of each iteration.
			8. At the end of the train call.
		:param device: Device to use for the training. Default is the device of the model.
		:param verbose: Whether to print information during training.
		:param kwargs: Additional arguments of the training.

		:Keyword Arguments:
			* <n_epochs>: int -> The number of epochs to train at each iteration. Default is 1.
			* <lr>: float -> Learning rate of the main optimizer. Default is 1e-3.
			* <reg_lr>: float -> Learning rate of the regularization optimizer. Default is 1e-2.
			* <weight_decay>: float -> Weight decay of the main optimizer. Default is 0.0.
			* <exec_metrics_on_train>: float -> Whether to compute metrics on the train dataset. This is useful when
			you want to save time by not computing the metrics on the train dataset. Default is True.
		"""
		assert model.is_built, "Model must be built before training"
		self.kwargs = self._set_default_kwargs(kwargs)
		self.model = model
		self.criterion = self._set_default_criterion(criterion)
		self.regularization = self._set_default_regularization(regularization)
		self.optimizer = self._set_default_optimizer(optimizer)
		self.regularization_optimizer = self._set_default_reg_optimizer(regularization_optimizer)
		self.metrics = self._set_default_metrics(metrics)
		self.callbacks: CallbacksList = self._set_default_callbacks(callbacks)
		self.sort_callbacks_()
		self.device = self._set_default_device(device)
		self.verbose = verbose
		self.training_history: TrainingHistory = self.training_histories[0]
		self.current_training_state = CurrentTrainingState.get_null_state()

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
	def load_checkpoint_mode(self):
		return self._load_checkpoint_mode

	@property
	def force_overwrite(self):
		return self._force_overwrite

	@property
	def training_histories(self) -> CallbacksList:
		return CallbacksList(list(filter(lambda x: isinstance(x, TrainingHistory), self.callbacks)))
	
	@property
	def checkpoint_managers(self) -> CallbacksList:
		return CallbacksList(list(filter(lambda x: isinstance(x, CheckpointManager), self.callbacks)))
	
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
		if optimizer is None:
			optimizer = torch.optim.Adam(
				self.model.parameters(),
				lr=self.kwargs["lr"],
				weight_decay=self.kwargs["weight_decay"],
			)
		return optimizer

	def _set_default_reg_optimizer(self, optimizer: Optional[torch.optim.Optimizer]) -> torch.optim.Optimizer:
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
		if criterion is None:
			if isinstance(self.model.output_sizes, dict):
				criterion = {
					k: torch.nn.MSELoss() for k in self.model.output_sizes
				}
			elif isinstance(self.model.output_sizes, int):
				criterion = torch.nn.MSELoss()
			else:
				raise ValueError("Unknown criterion type")
		return criterion

	def _set_default_regularization(
			self,
			regularization: Optional[Union[BaseRegularization, RegularizationList, Iterable[BaseRegularization]]]
	) -> Optional[RegularizationList]:
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
			device = self.model.device
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
		# TODO: sort by priority
		# histories = list(filter(lambda c: isinstance(c, TrainingHistory), self.callbacks))
		# checkpoints_mangers = list(filter(lambda c: isinstance(c, CheckpointManager), self.callbacks))
		# others = list(filter(lambda c: not isinstance(c, (TrainingHistory, CheckpointManager)), self.callbacks))
		# if reverse:
		# 	self.callbacks = CallbacksList(checkpoints_mangers + others + histories)
		# else:
		# 	self.callbacks = CallbacksList(histories + others + checkpoints_mangers)
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
			load_checkpoint_mode: LoadCheckpointMode = None,
			force_overwrite: bool = False,
			p_bar_position: Optional[int] = None,
			p_bar_leave: Optional[bool] = None,
			**kwargs
	) -> TrainingHistory:
		self._load_checkpoint_mode = load_checkpoint_mode
		self._force_overwrite = force_overwrite
		self.kwargs.update(kwargs)
		if n_iterations is None:
			n_iterations = self.kwargs["n_epochs"]
		self.sort_callbacks_()
		self.callbacks.start(self)
		self.load_state()
		if self.current_training_state.iteration is None:
			self.current_training_state = self.current_training_state.update(iteration=0)
		p_bar = tqdm(
			range(self.current_training_state.iteration, n_iterations),
			initial=self.current_training_state.iteration,
			total=n_iterations,
			desc=kwargs.get("desc", "Training"),
			disable=not self.verbose,
			position=p_bar_position,
			unit="itr",
			leave=p_bar_leave
		)
		for i in p_bar:
			self.current_training_state = self.current_training_state.update(iteration=i)
			self.callbacks.on_iteration_begin(self)
			itr_loss = self._exec_iteration(train_dataloader, val_dataloader)
			if self.kwargs["exec_metrics_on_train"]:
				itr_train_metrics = self._exec_metrics(train_dataloader, prefix="train")
			else:
				itr_train_metrics = {}
			if val_dataloader is not None:
				itr_val_metrics = self._exec_metrics(val_dataloader, prefix="val")
			else:
				itr_val_metrics = {}
			itr_metrics = dict(**itr_loss, **itr_train_metrics, **itr_val_metrics)
			postfix = {f"{k}": f"{v:.5e}" for k, v in itr_metrics.items()}
			self.current_training_state = self.current_training_state.update(itr_metrics=itr_metrics)
			self.callbacks.on_iteration_end(self)
			p_bar.set_postfix(postfix)
			if self.current_training_state.stop_training_flag:
				p_bar.set_postfix(OrderedDict(**{"stop_flag": "True"}, **postfix))
				break
		self.callbacks.close(self)
		p_bar.close()
		return self.training_history

	def _exec_iteration(
			self,
			train_dataloader: DataLoader,
			val_dataloader: Optional[DataLoader] = None
	) -> Dict[str, float]:
		with torch.no_grad():
			torch.cuda.empty_cache()
		losses = {}

		self.callbacks.on_train_begin(self)
		self.model.train()
		self.current_training_state = self.current_training_state.update(batch_is_train=True)
		train_loss = self._exec_epoch(train_dataloader)
		self.current_training_state = self.current_training_state.update(train_loss=train_loss)
		self.callbacks.on_train_end(self)
		losses["train_loss"] = train_loss

		if val_dataloader is not None:
			with torch.no_grad():
				self.callbacks.on_validation_begin(self)
				self.model.eval()
				self.current_training_state = self.current_training_state.update(batch_is_train=False)
				val_loss = self._exec_epoch(val_dataloader)
				self.current_training_state = self.current_training_state.update(val_loss=val_loss)
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
		for i, (x_batch, y_batch) in enumerate(dataloader):
			self.current_training_state = self.current_training_state.update(batch=i)
			batch_loss = self._exec_batch(x_batch, y_batch)
			batch_losses.append(batch_loss)
		mean_loss = np.mean(batch_losses)
		self.callbacks.on_epoch_end(self)
		return mean_loss

	def _exec_batch(
			self,
			x_batch,
			y_batch,
	):
		self.callbacks.on_batch_begin(self)
		x_batch = self._batch_to_dense(self._batch_to_device(x_batch))
		y_batch = self._batch_to_dense(self._batch_to_device(y_batch))
		batch_loss = self.apply_criterion_on_batch(x_batch, y_batch)
		if (
				hasattr(self.model, "get_and_reset_regularization_loss")
				and callable(self.model.get_and_reset_regularization_loss)
		):
			aux_regularization_loss = self.model.get_and_reset_regularization_loss()
			batch_loss += aux_regularization_loss
		if self.regularization_optimizer is None and self.regularization is not None:
			regularization_loss = self.regularization()
			batch_loss += regularization_loss
		if self.model.training:
			self.model.zero_grad()
			self.optimizer.zero_grad()
			batch_loss.backward()
			self.optimizer.step()
		if self.regularization_optimizer is not None and self.regularization is not None:
			regularization_loss = self.regularization()
			self.model.zero_grad()
			self.regularization_optimizer.zero_grad()
			regularization_loss.backward()
			self.regularization_optimizer.step()
		self.current_training_state = self.current_training_state.update(batch_loss=batch_loss.item())
		self.callbacks.on_batch_end(self)
		return batch_loss.item()

	def apply_criterion_on_batch(
			self,
			x_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
			y_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
	) -> torch.Tensor:
		if self.model.training:
			out = self.model(x_batch)
		else:
			with torch.no_grad():
				out = self.model(x_batch)
		
		if isinstance(out, (tuple, list)):
			pred = out[0]
		elif isinstance(out, torch.Tensor):
			pred = out
		else:
			raise ValueError(f"Unsupported output type: {type(out)}")
		
		if isinstance(self.criterion, dict):
			if isinstance(pred, dict) and len(pred) == 1 and len(self.criterion) == 1:
				pred = pred[list(pred.keys())[0]]
			
			if len(self.criterion) == 1 and isinstance(pred, torch.Tensor) and isinstance(y_batch, torch.Tensor):
				return list(self.criterion.values())[0](pred, y_batch.to(self.device))
			assert isinstance(x_batch, dict) and isinstance(y_batch, dict) and isinstance(pred, dict), \
				"If criterion is a dict, x_batch, y_batch and pred must be a dict too."
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


