from typing import Optional, List, Callable

import torch

from ..modules import BaseModel


class BaseTrainer:
	def __init__(
			self,
			model: BaseModel,
			loss_fn: torch.nn.Module = torch.nn.MSELoss(),
			optimizer: Optional[torch.optim.Optimizer] = None,
			metrics: Optional[List[Callable]] = None,
			callbacks: Optional[List[Callable]] = None,
			device: Optional[torch.device] = None,
			**kwargs
	):
		self.model = model
		self.loss_fn = loss_fn
		self.optimizer = optimizer
		self.metrics = metrics
		self.callbacks = callbacks
		self.device = device
		self.kwargs = kwargs

	def train(self, train_loader, val_loader, epochs=1, **kwargs):
		raise NotImplementedError()
	
	def _check_early_stopping(self, patience: int, tol: float = 1e-2) -> bool:
		"""
		:param patience:
		:return:
		"""
		losses = self.loss_history['val'][-patience:]
		return np.all(np.abs(np.diff(losses)) < tol)

	def fit(
			self,
			train_dataloader: DataLoader,
			val_dataloader: DataLoader,
			lr=1e-3,
			nb_epochs=15,
			criterion=None,
			optimizer=None,
			load_checkpoint_mode: LoadCheckpointMode = None,
			force_overwrite: bool = False,
			early_stopping: bool = False,
			early_stopping_patience: int = 5,
			verbose: bool = True,
			p_bar_position: Optional[int] = None,
			p_bar_leave: Optional[bool] = None,
	):
		if criterion is None:
			criterion = nn.NLLLoss()
		if optimizer is None:
			optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)

		start_epoch = 0
		if load_checkpoint_mode is None:
			assert os.path.exists(self.checkpoints_meta_path) or force_overwrite, \
				f"{self.checkpoints_meta_path} already exists. " \
				f"Set force_overwrite flag to True to overwrite existing saves."
			if os.path.exists(self.checkpoints_meta_path) and force_overwrite:
				shutil.rmtree(self.checkpoint_folder)
		else:
			try:
				checkpoint = self.load_checkpoint(load_checkpoint_mode)
				self.load_state_dict(checkpoint[SequentialModel.CHECKPOINT_STATE_DICT_KEY], strict=True)
				optimizer.load_state_dict(checkpoint[SequentialModel.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY])
				start_epoch = int(checkpoint[SequentialModel.CHECKPOINT_EPOCH_KEY]) + 1
				self.loss_history = self.get_checkpoints_loss_history()
			except FileNotFoundError:
				if verbose:
					logging.warning("No such checkpoint. Fit from beginning.", UserWarning)

		if start_epoch >= nb_epochs:
			return self.loss_history

		best_loss = self.loss_history.min('val')
		p_bar = tqdm(
			range(start_epoch, nb_epochs),
			desc="Training",
			disable=not verbose,
			position=p_bar_position,
			unit="epoch",
			leave=p_bar_leave
		)
		for epoch in p_bar:
			epoch_loss = self._exec_phase(train_dataloader, val_dataloader, criterion, optimizer)
			epoch_val_acc = self.compute_classification_accuracy(val_dataloader, verbose=False)
			self.loss_history.concat(epoch_loss)
			is_best = epoch_loss['val'] < best_loss
			self.save_checkpoint(optimizer, epoch, epoch_loss, is_best)
			if is_best:
				best_loss = epoch_loss['val']
			p_bar.set_postfix(
				train_loss=f"{epoch_loss['train']:.5e}",
				val_loss=f"{epoch_loss['val']:.5e}",
				val_acc=f"{epoch_val_acc:.5f}",
			)
			if early_stopping and self._check_early_stopping(early_stopping_patience):
				if verbose:
					logging.info(f"Early stopping stopped the training at epoch {epoch}.")
				break
		p_bar.close()
		self.plot_loss_history(show=False)
		return self.loss_history

	def _exec_phase(self, train_dataloader, val_dataloader, criterion, optimizer):
		self.train()
		train_loss = self._exec_epoch(
			train_dataloader,
			criterion,
			optimizer,
		)
		self.eval()
		val_loss = self._exec_epoch(
			val_dataloader,
			criterion,
			optimizer,
		)
		return dict(train=train_loss, val=val_loss)

	def _exec_epoch(
			self,
			dataloader,
			criterion,
			optimizer,
	):
		batch_losses = []
		for x_batch, y_batch in dataloader:
			batch_loss = self._exec_batch(
				x_batch,
				y_batch,
				criterion,
				optimizer,
			)
			batch_losses.append(batch_loss)
		return np.mean(batch_losses)

	def _exec_batch(
			self,
			x_batch,
			y_batch,
			criterion,
			optimizer,
	):
		if self.training:
			log_p_y, out, h_sates = self.get_prediction_log_proba(
				x_batch, re_outputs_trace=True, re_hidden_states=True
			)
		else:
			with torch.no_grad():
				log_p_y, out, h_sates = self.get_prediction_log_proba(
					x_batch, re_outputs_trace=True, re_hidden_states=True
				)

		# TODO: add regularization loss
		# reg_loss = torch.mean(self.get_spikes_count_per_neuron(h_sates))
		# spikes = [h[-1] for l_name, h_list in h_sates.items() for h in h_list if l_name.lower() != "readout"]
		# reg_loss = 1e-5 * sum([torch.sum(s) for s in spikes])  # L1 loss on total number of spikes
		# reg_loss = 1e-5 * sum(
		# 	[torch.mean(torch.sum(torch.sum(s, dim=0), dim=0) ** 2) for s in spikes]
		# )  # L2 loss on spikes per neuron
		# reg_loss = torch.mean(self.get_spikes_count_per_neuron(h_sates) ** 2)

		batch_loss = criterion(log_p_y, y_batch.long().to(self.device))
		if self.training:
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()
		return batch_loss.item()




