from typing import List, Iterable, Optional


class BaseCallback:
	def start(self, trainer):
		pass
	
	def close(self, trainer):
		pass
	
	def on_train_begin(self, trainer):
		pass
	
	def on_train_end(self, trainer):
		pass
	
	def on_epoch_begin(self, trainer):
		pass
	
	def on_epoch_end(self, trainer):
		pass
	
	def on_batch_begin(self, trainer):
		pass
	
	def on_batch_end(self, trainer):
		pass
	
	def on_validation_begin(self, trainer):
		pass
	
	def on_validation_end(self, trainer):
		pass

	def on_iteration_end(self, trainer):
		pass

	
class CallbacksList:
	def __init__(self, callbacks: Optional[Iterable[BaseCallback]] = None):
		if callbacks is None:
			callbacks = []
		assert isinstance(callbacks, Iterable), "callbacks must be an Iterable"
		assert all(isinstance(callback, BaseCallback) for callback in callbacks), \
			"All callbacks must be instances of BaseCallback"
		self.callbacks = callbacks
		self._length = len([_ for _ in self.callbacks])
		
	def __iter__(self):
		return iter(self.callbacks)
	
	def __len__(self):
		return self._length
	
	def start(self, trainer):
		for callback in self.callbacks:
			callback.start(trainer)
	
	def close(self, trainer):
		for callback in self.callbacks:
			callback.close(trainer)
	
	def on_train_begin(self, trainer):
		for callback in self.callbacks:
			callback.on_train_begin(trainer)
	
	def on_train_end(self, trainer):
		for callback in self.callbacks:
			callback.on_train_end(trainer)
	
	def on_epoch_begin(self, trainer):
		for callback in self.callbacks:
			callback.on_epoch_begin(trainer)
	
	def on_epoch_end(self, trainer):
		for callback in self.callbacks:
			callback.on_epoch_end(trainer)
	
	def on_batch_begin(self, trainer):
		for callback in self.callbacks:
			callback.on_batch_begin(trainer)
	
	def on_batch_end(self, trainer):
		for callback in self.callbacks:
			callback.on_batch_end(trainer)
	
	def on_validation_begin(self, trainer):
		for callback in self.callbacks:
			callback.on_validation_begin(trainer)
	
	def on_validation_end(self, trainer):
		for callback in self.callbacks:
			callback.on_validation_end(trainer)

	def on_iteration_end(self, trainer):
		for callback in self.callbacks:
			callback.on_iteration_end(trainer)
