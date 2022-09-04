from typing import Iterable, Optional, Iterator


class BaseCallback:
	def start(self, trainer):
		"""
		Called when the training starts. This is the first callback called.
		
		:param trainer: The trainer.
		:type trainer: Trainer
		
		:return: None
		"""
		pass
	
	def close(self, trainer):
		"""
		Called when the training ends. This is the last callback called.
		
		:param trainer: The trainer.
		:type trainer: Trainer
		
		:return: None
		"""
		pass

	def on_train_begin(self, trainer):
		"""
		Called when the train phase of an iteration starts. The train phase is defined as a full pass through the
		training dataset.
		
		:param trainer: The trainer.
		:type trainer: Trainer
		
		:return: None
		"""
		pass

	def on_train_end(self, trainer):
		"""
		Called when the train phase of an iteration ends. The train phase is defined as a full pass through the
		training dataset.
		
		:param trainer: The trainer.
		:type trainer: Trainer
		
		:return: None
		"""
		pass

	def on_epoch_begin(self, trainer):
		"""
		Called when an epoch starts. An epoch is defined as one full pass through the training dataset or
		the validation dataset.
		
		:param trainer: The trainer.
		:type trainer: Trainer
		
		:return: None
		"""
		pass

	def on_epoch_end(self, trainer):
		"""
		Called when an epoch ends. An epoch is defined as one full pass through the training dataset or
		the validation dataset.
		
		:param trainer: The trainer.
		:type trainer: Trainer
		
		:return: None
		"""
		pass

	def on_batch_begin(self, trainer):
		"""
		Called when a batch starts. The batch is defined as one forward pass through the network.
		
		:param trainer: The trainer.
		:type trainer: Trainer
		
		:return: None
		"""
		pass

	def on_batch_end(self, trainer):
		"""
		Called when a batch ends. The batch is defined as one forward pass through the network.
		
		:param trainer: The trainer.
		:type trainer: Trainer
		
		:return: None
		"""
		pass

	def on_validation_begin(self, trainer):
		"""
		Called when the validation phase of an iteration starts. The validation phase is defined as a full pass through
		the validation dataset.
		
		:param trainer: The trainer.
		:type trainer: Trainer
		
		:return:
		"""
		pass

	def on_validation_end(self, trainer):
		"""
		Called when the validation phase of an iteration ends. The validation phase is defined as a full pass through
		the validation dataset.
		
		:param trainer: The trainer.
		:type trainer: Trainer
		
		:return: None
		"""
		pass

	def on_iteration_begin(self, trainer):
		"""
		Called when an iteration starts. An iteration is defined as one full pass through the training dataset and
		the validation dataset.
		
		:param trainer: The trainer.
		:type trainer: Trainer
		
		:return: None
		"""
		pass

	def on_iteration_end(self, trainer):
		"""
		Called when an iteration ends. An iteration is defined as one full pass through the training dataset and
		the validation dataset.
		
		:param trainer: The trainer.
		:type trainer: Trainer
		
		:return: None
		"""
		pass

	
class CallbacksList:
	"""
	This class is used to store the callbacks that are used during the training. Each callback of the list is called
	in the order they are stored in the list.
	
	:Attributes:
		- **callbacks** (List[BaseCallback]): The callbacks to use.
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
	
	def start(self, trainer):
		"""
		Called when the trainer starts.
		
		:param trainer: The trainer.
		:type trainer: Trainer
		
		:return: None
		"""
		for callback in self.callbacks:
			callback.start(trainer)
	
	def close(self, trainer):
		"""
		Called when the trainer closes.
		
		:param trainer: The trainer.
		:type trainer: Trainer
		
		:return: None
		"""
		for callback in self.callbacks:
			callback.close(trainer)
	
	def on_train_begin(self, trainer):
		"""
		Called when the train phase of an iteration starts. The train phase is defined as a full pass through the
		training dataset.
		
		:param trainer: The trainer.
		:type trainer: Trainer
		
		:return: None
		"""
		for callback in self.callbacks:
			callback.on_train_begin(trainer)
	
	def on_train_end(self, trainer):
		"""
		Called when the train phase of an iteration ends. The train phase is defined as a full pass through the
		training dataset.
		
		:param trainer: The trainer.
		:type trainer: Trainer
		
		:return: None
		"""
		for callback in self.callbacks:
			callback.on_train_end(trainer)
	
	def on_epoch_begin(self, trainer):
		"""
		Called when an epoch starts. An epoch is defined as one full pass through the training dataset or
		the validation dataset.
		
		:param trainer: The trainer.
		:type trainer: Trainer
		
		:return: None
		"""
		for callback in self.callbacks:
			callback.on_epoch_begin(trainer)
	
	def on_epoch_end(self, trainer):
		"""
		Called when an epoch ends. An epoch is defined as one full pass through the training dataset or
		the validation dataset.
		
		:param trainer: The trainer.
		:type trainer: Trainer
		
		:return: None
		"""
		for callback in self.callbacks:
			callback.on_epoch_end(trainer)
	
	def on_batch_begin(self, trainer):
		"""
		Called when a batch starts. The batch is defined as one forward pass through the network.
		
		:param trainer: The trainer.
		:type trainer: Trainer
		
		:return: None
		"""
		for callback in self.callbacks:
			callback.on_batch_begin(trainer)
	
	def on_batch_end(self, trainer):
		"""
		Called when a batch ends. The batch is defined as one forward pass through the network.
		
		:param trainer: The trainer.
		:type trainer: Trainer
		
		:return: None
		"""
		for callback in self.callbacks:
			callback.on_batch_end(trainer)
	
	def on_validation_begin(self, trainer):
		"""
		Called when the validation phase of an iteration starts. The validation phase is defined as a full pass through
		the validation dataset.
		
		:param trainer: The trainer.
		:type trainer: Trainer
		
		:return: None
		"""
		for callback in self.callbacks:
			callback.on_validation_begin(trainer)
	
	def on_validation_end(self, trainer):
		"""
		Called when the validation phase of an iteration ends. The validation phase is defined as a full pass through
		the validation dataset.
		
		:param trainer: The trainer.
		:type trainer: Trainer
		
		:return: None
		"""
		for callback in self.callbacks:
			callback.on_validation_end(trainer)

	def on_iteration_begin(self, trainer):
		"""
		Called when an iteration starts. An iteration is defined as one full pass through the training dataset and
		the validation dataset.
		
		:param trainer: The trainer.
		:type trainer: Trainer
		
		:return: None
		"""
		for callback in self.callbacks:
			callback.on_iteration_begin(trainer)

	def on_iteration_end(self, trainer):
		"""
		Called when an iteration ends. An iteration is defined as one full pass through the training dataset and
		the validation dataset.
		
		:param trainer: The trainer.
		:type trainer: Trainer
		
		:return: None
		"""
		for callback in self.callbacks:
			callback.on_iteration_end(trainer)

