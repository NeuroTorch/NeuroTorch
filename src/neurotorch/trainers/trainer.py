from typing import Optional, List, Callable, Dict, Any, Union, NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..callbacks import LoadCheckpointMode, TrainingHistory
from ..callbacks.base_callback import BaseCallback, CallbacksList
from ..modules import BaseModel


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
		)


class Trainer:
	def __init__(
			self,
			model: BaseModel,
			criterion: Optional[Union[Dict[str, torch.nn.Module], torch.nn.Module]] = None,
			optimizer: Optional[torch.optim.Optimizer] = None,
			metrics: Optional[List[Callable]] = None,
			callbacks: Optional[Union[List[BaseCallback], CallbacksList, BaseCallback]] = None,
			device: Optional[torch.device] = None,
			verbose: bool = True,
			**kwargs
	):
		self.kwargs = self._set_default_kwargs(kwargs)
		self.model = model
		self.criterion = self._set_default_criterion(criterion)
		self.optimizer = self._set_default_optimizer(optimizer)
		self.metrics = self._set_default_metrics(metrics)
		self.callbacks: CallbacksList = self._set_default_callbacks(callbacks)
		self.device = self._set_default_device(device)
		self.verbose = verbose
		self.training_history = list(filter(lambda x: isinstance(x, TrainingHistory), self.callbacks.callbacks))[0]
		self.current_training_state = CurrentTrainingState.get_null_state()

		self._load_checkpoint_mode = None
		self._force_overwrite = None

	@property
	def load_checkpoint_mode(self):
		return self._load_checkpoint_mode

	@property
	def force_overwrite(self):
		return self._force_overwrite
	
	@staticmethod
	def _set_default_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
		kwargs.setdefault("n_epochs", 3)
		kwargs.setdefault("lr", 1e-3)
		kwargs.setdefault("weight_decay", 1e-5)
		kwargs.setdefault("batch_size", 256)
		
		assert kwargs["batch_size"] > 0, "batch_size must be positive"
		return kwargs
	
	def _set_default_optimizer(self, optimizer: Optional[torch.optim.Optimizer]) -> torch.optim.Optimizer:
		if optimizer is None:
			optimizer = torch.optim.Adam(
				self.model.parameters(),
				lr=self.kwargs["lr"],
				weight_decay=self.kwargs["weight_decay"]
			)
		return optimizer

	def _set_default_metrics(self, metrics: Optional[List[Callable]]):
		if metrics is None:
			metrics = [lambda *x: None]
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

	def train(
			self,
			train_dataloader: DataLoader,
			val_dataloader: DataLoader,
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
		self.callbacks.start(self)
		if self.current_training_state.iteration is None:
			self.current_training_state = self.current_training_state.update(iteration=0)
		p_bar = tqdm(
			range(self.current_training_state.iteration, n_iterations),
			desc="Training",
			disable=not self.verbose,
			position=p_bar_position,
			unit="itr",
			leave=p_bar_leave
		)
		for i in p_bar:
			self.current_training_state = self.current_training_state.update(iteration=i)
			itr_loss = self._exec_iteration(train_dataloader, val_dataloader)
			itr_val_metrics = self._exec_metrics(val_dataloader)
			itr_metrics = dict(**itr_loss, **itr_val_metrics)
			postfix = dict(
				train_loss=f"{itr_loss['train_loss']:.5e}",
				val_loss=f"{itr_loss['val_loss']:.5e}",
			)
			postfix.update({f"val_{k}": f"{v:.5e}" for k, v in itr_val_metrics.items()})
			self.current_training_state = self.current_training_state.update(itr_metrics=itr_metrics)
			self.callbacks.on_iteration_end(self)
			p_bar.set_postfix(postfix)
		self.callbacks.close(self)
		p_bar.close()
		# self.plot_loss_history(show=False)
		return self.training_history

	def _exec_iteration(self, train_dataloader, val_dataloader):
		self.callbacks.on_train_begin(self)
		self.model.train()
		self.current_training_state = self.current_training_state.update(batch_is_train=True)
		train_loss = self._exec_epoch(train_dataloader)
		self.current_training_state = self.current_training_state.update(train_loss=train_loss)
		self.callbacks.on_train_end(self)
		
		self.callbacks.on_validation_begin(self)
		self.model.eval()
		self.current_training_state = self.current_training_state.update(batch_is_train=False)
		val_loss = self._exec_epoch(val_dataloader)
		self.current_training_state = self.current_training_state.update(val_loss=val_loss)
		self.callbacks.on_validation_end(self)
		return dict(train_loss=train_loss, val_loss=val_loss)

	def _exec_metrics(self, dataloader) -> Dict:
		metrics_dict = {}
		for metric in self.metrics:
			m_out = metric(dataloader)
			if isinstance(m_out, dict):
				metrics_dict.update(m_out)
			else:
				metrics_dict[metric.__name__] = m_out
		return metrics_dict

	def _exec_epoch(
			self,
			dataloader,
	):
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
		batch_loss = self.apply_criterion_on_batch(x_batch, y_batch)
		if self.model.training:
			self.optimizer.zero_grad()
			batch_loss.backward()
			self.optimizer.step()
		self.current_training_state = self.current_training_state.update(batch_loss=batch_loss.item())
		self.callbacks.on_batch_end(self)
		return batch_loss.item()

	def apply_criterion_on_batch(self, x_batch, y_batch):
		if self.model.training:
			pred, out, h_sates = self.model.get_raw_prediction(
				x_batch, re_outputs_trace=True, re_hidden_states=True
			)
		else:
			with torch.no_grad():
				pred, out, h_sates = self.model.get_raw_prediction(
					x_batch, re_outputs_trace=True, re_hidden_states=True
				)
		if isinstance(self.criterion, dict):
			if len(self.criterion) == 1 and isinstance(pred, torch.Tensor) and isinstance(y_batch, torch.Tensor):
				return list(self.criterion.values())[0](pred, y_batch.long().to(self.device))
			assert isinstance(x_batch, dict) and isinstance(y_batch, dict) and isinstance(pred, dict), \
				"If criterion is a dict, x_batch, y_batch and pred must be a dict too."
			batch_loss = sum([
				self.criterion[k](pred[k], y_batch[k].long().to(self.device))
				for k in self.criterion
			])
		else:
			batch_loss = self.criterion(pred, y_batch.long().to(self.device))
		return batch_loss

