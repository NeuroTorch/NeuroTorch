from collections import defaultdict

import neurotorch as nt
from neurotorch.callbacks.base_callback import CallbacksList, BaseCallback
from neurotorch.trainers.trainer import CurrentTrainingState


class MockHistory(BaseCallback):
	def __init__(self, ins_id=0, **kwargs):
		super().__init__(**kwargs)
		self.ins_id = ins_id
		self.min_call_flag = False
		self.max_call_flag = False
	
	def reset_mock(self):
		self.min_call_flag = False
		self.max_call_flag = False
	
	def min(self, *args, **kwargs):
		self.min_call_flag = True
	
	def max(self, *args, **kwargs):
		self.max_call_flag = True
	
	def __eq__(self, other):
		return self.ins_id == other.ins_id
	
	def __repr__(self):
		return f"<History{self.ins_id}>"
	
	def plot(self, *args, **kwargs):
		pass


class MockTrainer:
	def __init__(self):
		self.training_history = MockHistory()
		self.callbacks = CallbacksList([self.training_history])
		self.sort_flag = False
		self.load_checkpoint_mode = None
		self.force_overwrite = False
		self.current_training_state = CurrentTrainingState()
		self.model = nt.SequentialModel(layers=[nt.LIFLayer(10, 10)]).build()
		self.optimizer = None
	
	def sort_callbacks_(self):
		self.sort_flag = True
		self.callbacks.sort_callbacks_()
		
	def train(self, *args, **kwargs):
		n_iterations = kwargs.get("n_iterations", 1)
		self.sort_callbacks_()
		self.callbacks.start(self)
		self.callbacks.load_checkpoint_state(self, {})
		for i in range(n_iterations):
			self.current_training_state = self.current_training_state.update(iteration=i)
			self.callbacks.on_iteration_begin(self)
			self.callbacks.on_train_begin(self)
			self.callbacks.on_epoch_begin(self)
			self.callbacks.on_batch_begin(self)
			self.callbacks.on_batch_end(self)
			self.callbacks.on_epoch_end(self)
			self.callbacks.on_train_end(self)
			self.callbacks.on_validation_begin(self)
			self.callbacks.on_epoch_begin(self)
			self.callbacks.on_batch_begin(self)
			self.callbacks.on_batch_end(self)
			self.callbacks.on_epoch_end(self)
			self.callbacks.on_validation_end(self)
			self.current_training_state = self.current_training_state.update(itr_metrics={})
			self.callbacks.on_iteration_end(self)
		self.callbacks.close(self)
		
	def update_state_(self, **kwargs):
		self.current_training_state = self.current_training_state.update(**kwargs)


class MockCallback(BaseCallback):
	def __init__(self, **kwargs):
		super(MockCallback, self).__init__(**kwargs)
		self.call_mthds_counter = defaultdict(int)
		
	def start(self, trainer):
		self.call_mthds_counter['start'] += 1
	
	def on_iteration_begin(self, trainer):
		self.call_mthds_counter['on_iteration_begin'] += 1
		
	def on_train_begin(self, trainer):
		self.call_mthds_counter['on_train_begin'] += 1
		
	def on_epoch_begin(self, trainer):
		self.call_mthds_counter['on_epoch_begin'] += 1
		
	def on_batch_begin(self, trainer):
		self.call_mthds_counter['on_batch_begin'] += 1
		
	def on_batch_end(self, trainer):
		self.call_mthds_counter['on_batch_end'] += 1
	
	def on_epoch_end(self, trainer):
		self.call_mthds_counter['on_epoch_end'] += 1
	
	def on_train_end(self, trainer):
		self.call_mthds_counter['on_train_end'] += 1
		
	def on_validation_begin(self, trainer):
		self.call_mthds_counter['on_validation_begin'] += 1
		
	def on_validation_end(self, trainer):
		self.call_mthds_counter['on_validation_end'] += 1
		
	def on_iteration_end(self, trainer):
		self.call_mthds_counter['on_iteration_end'] += 1
		
	def close(self, trainer):
		self.call_mthds_counter['close'] += 1
		
	def load_checkpoint_state(self, trainer, state):
		self.call_mthds_counter['load_checkpoint_state'] += 1
		
	def save_checkpoint_state(self, trainer):
		self.call_mthds_counter['save_checkpoint_state'] += 1
		return {}
	