from typing import Optional, List, Callable, Dict, Any, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..callbacks import LoadCheckpointMode, TrainingHistory
from ..callbacks.base_callback import BaseCallback, CallbacksList
from ..modules import BaseModel


class Trainer:
	def __init__(
			self,
			model: BaseModel,
			criterion: torch.nn.Module = torch.nn.MSELoss(),
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
		self.metrics = metrics
		self.callbacks = self._set_default_callbacks(callbacks)
		self.device = self._set_default_device(device)
		self.verbose = verbose
		self.training_history = TrainingHistory()
	
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
	
	@staticmethod
	def _set_default_criterion(criterion: Optional[torch.nn.Module]) -> torch.nn.Module:
		if criterion is None:
			criterion = torch.nn.MSELoss()
		return criterion
	
	def _set_default_device(self, device: Optional[torch.device]) -> torch.device:
		if device is None:
			device = self.model.device
		return device
	
	@staticmethod
	def _set_default_callbacks(
			callbacks: Optional[Union[List[BaseCallback], CallbacksList, BaseCallback]]
	) -> CallbacksList:
		if isinstance(callbacks, BaseCallback):
			callbacks = [callbacks]
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
	):
		self.kwargs.update(kwargs)
		start_epoch = 0
		if n_iterations is None:
			n_iterations = self.kwargs["n_epochs"]
		# if load_checkpoint_mode is None:
		# 	assert os.path.exists(self.checkpoints_meta_path) or force_overwrite, \
		# 		f"{self.checkpoints_meta_path} already exists. " \
		# 		f"Set force_overwrite flag to True to overwrite existing saves."
		# 	if os.path.exists(self.checkpoints_meta_path) and force_overwrite:
		# 		shutil.rmtree(self.checkpoint_folder)
		# else:
		# 	try:
		# 		checkpoint = self.load_checkpoint(load_checkpoint_mode)
		# 		self.load_state_dict(checkpoint[SequentialModel.CHECKPOINT_STATE_DICT_KEY], strict=True)
		# 		optimizer.load_state_dict(checkpoint[SequentialModel.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY])
		# 		start_epoch = int(checkpoint[SequentialModel.CHECKPOINT_EPOCH_KEY]) + 1
		# 		self.loss_history = self.get_checkpoints_loss_history()
		# 	except FileNotFoundError:
		# 		if verbose:
		# 			logging.warning("No such checkpoint. Fit from beginning.", UserWarning)
		#
		# if start_epoch >= nb_epochs:
		# 	return self.loss_history

		# best_loss = self.loss_history.min('val')
		p_bar = tqdm(
			range(start_epoch, n_iterations),
			desc="Training",
			disable=not self.verbose,
			position=p_bar_position,
			unit="epoch",
			leave=p_bar_leave
		)
		self.callbacks.start(self)
		for i in p_bar:
			itr_loss = self._exec_iteration(train_dataloader, val_dataloader)
			# self.loss_history.concat(epoch_loss)
			# is_best = epoch_loss['val'] < best_loss
			# self.save_checkpoint(optimizer, epoch, epoch_loss, is_best)
			# if is_best:
			# 	best_loss = epoch_loss['val']
			p_bar.set_postfix(
				train_loss=f"{itr_loss['train']:.5e}",
				val_loss=f"{itr_loss['val']:.5e}",
			)
		self.callbacks.close(self)
		p_bar.close()
		# self.plot_loss_history(show=False)
		# return self.loss_history

	def _exec_iteration(self, train_dataloader, val_dataloader):
		self.callbacks.on_train_begin(self)
		self.model.train()
		train_loss = self._exec_epoch(
			train_dataloader,
		)
		self.callbacks.on_train_end(self)
		self.callbacks.on_validation_begin(self)
		self.model.eval()
		val_loss = self._exec_epoch(
			val_dataloader,
		)
		self.callbacks.on_validation_end(self)
		return dict(train=train_loss, val=val_loss)

	def _exec_epoch(
			self,
			dataloader,
	):
		self.callbacks.on_epoch_begin(self)
		batch_losses = []
		for x_batch, y_batch in dataloader:
			batch_loss = self._exec_batch(
				x_batch,
				y_batch,
			)
			batch_losses.append(batch_loss)
		self.callbacks.on_epoch_end(self)
		return np.mean(batch_losses)

	def _exec_batch(
			self,
			x_batch,
			y_batch,
	):
		self.callbacks.on_batch_begin(self)
		if self.model.training:
			log_p_y, out, h_sates = self.model.get_prediction_log_proba(
				x_batch, re_outputs_trace=True, re_hidden_states=True
			)
		else:
			with torch.no_grad():
				log_p_y, out, h_sates = self.model.get_prediction_log_proba(
					x_batch, re_outputs_trace=True, re_hidden_states=True
				)
		
		batch_loss = self.criterion(log_p_y, y_batch.long().to(self.device))
		if self.model.training:
			self.optimizer.zero_grad()
			batch_loss.backward()
			self.optimizer.step()
		self.callbacks.on_batch_end(self)
		return batch_loss.item()




