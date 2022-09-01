from typing import List, Iterable, Optional


class BaseCallback:
	def start(self, trainer):
		"""
		Called when the training starts. This is the first callback called.
		
		:param trainer: The trainer.
		:return: None
		"""
		pass
	
	def close(self, trainer):
		"""
		Called when the training ends. This is the last callback called.
		
		:param trainer: The trainer.
		:return: None
		"""
		pass

	def on_train_begin(self, trainer):
		"""
		Called when the train phase of an iteration starts. The train phase is defined as a full pass through the
		training dataset.
		
		:param trainer: The trainer.
		:return: None
		"""
		pass

	def on_train_end(self, trainer):
		"""
		Called when the train phase of an iteration ends. The train phase is defined as a full pass through the
		training dataset.
		
		:param trainer: The trainer.
		:return: None
		"""
		pass

	def on_epoch_begin(self, trainer):
		"""
		Called when an epoch starts. An epoch is defined as one full pass through the training dataset or
		the validation dataset.
		
		:param trainer: The trainer.
		:return: None
		"""
		pass

	def on_epoch_end(self, trainer):
		"""
		Called when an epoch ends. An epoch is defined as one full pass through the training dataset or
		the validation dataset.
		
		:param trainer: The trainer.
		:return: None
		"""
		pass

	def on_batch_begin(self, trainer):
		"""
		Called when a batch starts. The batch is defined as one forward pass through the network.
		
		:param trainer: The trainer.
		:return: None
		"""
		pass

	def on_batch_end(self, trainer):
		"""
		Called when a batch ends. The batch is defined as one forward pass through the network.
		
		:param trainer: The trainer.
		:return: None
		"""
		pass

	def on_validation_begin(self, trainer):
		"""
		Called when the validation phase of an iteration starts. The validation phase is defined as a full pass through
		the validation dataset.
		
		:param trainer:
		:return:
		"""
		pass

	def on_validation_end(self, trainer):
		"""
		Called when the validation phase of an iteration ends. The validation phase is defined as a full pass through
		the validation dataset.
		
		:param trainer: The trainer.
		:return: None
		"""
		pass

	def on_iteration_begin(self, trainer):
		"""
		Called when an iteration starts. An iteration is defined as one full pass through the training dataset and
		the validation dataset.
		
		:param trainer: The trainer.
		:return: None
		"""
		pass

	def on_iteration_end(self, trainer):
		"""
		Called when an iteration ends. An iteration is defined as one full pass through the training dataset and
		the validation dataset.
		
		:param trainer: The trainer.
		:return: None
		"""
		pass

	
class CallbacksList:
	def __init__(self, callbacks: Optional[Iterable[BaseCallback]] = None):
		if callbacks is None:
			callbacks = []
		assert isinstance(callbacks, Iterable), "callbacks must be an Iterable"
		assert all(isinstance(callback, BaseCallback) for callback in callbacks), \
			"All callbacks must be instances of BaseCallback"
		self.callbacks = list(callbacks)
		self._length = len(self.callbacks)

	def __getitem__(self, item):
		return self.callbacks[item]

	def __iter__(self):
		return iter(self.callbacks)
	
	def __len__(self):
		return self._length

	def append(self, callback: BaseCallback):
		"""
		Append a callback to the list.
		:param callback: The callback to append.
		:return: None
		"""
		assert isinstance(callback, BaseCallback), "callback must be an instance of BaseCallback"
		self.callbacks.append(callback)
		self._length += 1

	def remove(self, callback: BaseCallback):
		"""
		Remove a callback from the list.
		:param callback: The callback to remove.
		:return: None
		"""
		assert isinstance(callback, BaseCallback), "callback must be an instance of BaseCallback"
		self.callbacks.remove(callback)
		self._length -= 1
	
	def start(self, trainer):
		"""
		Called when the trainer starts.
		:param trainer: The trainer.
		:return: None
		"""
		for callback in self.callbacks:
			callback.start(trainer)
	
	def close(self, trainer):
		"""
		Called when the trainer closes.
		:param trainer: The trainer.
		:return: None
		"""
		for callback in self.callbacks:
			callback.close(trainer)
	
	def on_train_begin(self, trainer):
		"""
		Called when the train phase of an iteration starts. The train phase is defined as a full pass through the
		training dataset.
		:param trainer: The trainer.
		:return: None
		"""
		for callback in self.callbacks:
			callback.on_train_begin(trainer)
	
	def on_train_end(self, trainer):
		"""
		Called when the train phase of an iteration ends. The train phase is defined as a full pass through the
		training dataset.
		:param trainer: The trainer.
		:return: None
		"""
		for callback in self.callbacks:
			callback.on_train_end(trainer)
	
	def on_epoch_begin(self, trainer):
		"""
		Called when an epoch starts. An epoch is defined as one full pass through the training dataset or
		the validation dataset.
		:param trainer: The trainer.
		:return: None
		"""
		for callback in self.callbacks:
			callback.on_epoch_begin(trainer)
	
	def on_epoch_end(self, trainer):
		"""
		Called when an epoch ends. An epoch is defined as one full pass through the training dataset or
		the validation dataset.
		:param trainer: The trainer.
		:return: None
		"""
		for callback in self.callbacks:
			callback.on_epoch_end(trainer)
	
	def on_batch_begin(self, trainer):
		"""
		Called when a batch starts. The batch is defined as one forward pass through the network.
		:param trainer: The trainer.
		:return: None
		"""
		for callback in self.callbacks:
			callback.on_batch_begin(trainer)
	
	def on_batch_end(self, trainer):
		"""
		Called when a batch ends. The batch is defined as one forward pass through the network.
		:param trainer: The trainer.
		:return: None
		"""
		for callback in self.callbacks:
			callback.on_batch_end(trainer)
	
	def on_validation_begin(self, trainer):
		"""
		Called when the validation phase of an iteration starts. The validation phase is defined as a full pass through
		the validation dataset.
		:param trainer:
		:return:
		"""
		for callback in self.callbacks:
			callback.on_validation_begin(trainer)
	
	def on_validation_end(self, trainer):
		"""
		Called when the validation phase of an iteration ends. The validation phase is defined as a full pass through
		the validation dataset.
		:param trainer: The trainer.
		:return: None
		"""
		for callback in self.callbacks:
			callback.on_validation_end(trainer)

	def on_iteration_begin(self, trainer):
		"""
		Called when an iteration starts. An iteration is defined as one full pass through the training dataset and
		the validation dataset.
		:param trainer: The trainer.
		:return: None
		"""
		for callback in self.callbacks:
			callback.on_iteration_begin(trainer)

	def on_iteration_end(self, trainer):
		"""
		Called when an iteration ends. An iteration is defined as one full pass through the training dataset and
		the validation dataset.
		:param trainer: The trainer.
		:return: None
		"""
		for callback in self.callbacks:
			callback.on_iteration_end(trainer)

