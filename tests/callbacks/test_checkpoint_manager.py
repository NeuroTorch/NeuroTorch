import functools
import json
import os
import shutil
import unittest

import numpy as np
import torch

from neurotorch.callbacks import CheckpointManager, LoadCheckpointMode
from neurotorch.modules import LIFLayer, SequentialModel
from neurotorch.trainers.trainer import CurrentTrainingState


def _manage_temp_checkpoints_folder(_func=None, *, temp_folder: str = "./temp"):
	def decorator_log_func(func):
		@functools.wraps(func)
		def wrapper(*args, **kwargs):
			self = args[0]
			# clear the cache before running the test
			if os.path.exists(temp_folder):
				shutil.rmtree(temp_folder)
			try:
				out = func(*args, **kwargs)
			except Exception as e:
				self.fail(f"Exception raised: {e}")
			finally:
				if os.path.exists('./temp'):
					shutil.rmtree("./temp")
			return out
		
		wrapper.__name__ = func.__name__
		return wrapper
	
	if _func is None:
		return decorator_log_func
	else:
		return decorator_log_func(_func)


class MockHistory:
	def __init__(self, ins_id=0):
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


class MockTrainer:
	def __init__(self):
		self.training_history = MockHistory()
		self.callbacks = [self.training_history]
		self.sort_flag = False
		self.load_checkpoint_mode = None
		self.force_overwrite = False
		self.current_training_state = CurrentTrainingState.get_null_state()
		self.model = None
		self.optimizer = None
	
	def sort_callbacks_(self):
		self.sort_flag = True


class TestCheckpointManager(unittest.TestCase):
	@_manage_temp_checkpoints_folder
	def test_default_constructor(self):
		"""
		Test that the default constructor works as expected.
		:return: None
		"""
		checkpoint_manager = CheckpointManager()
		self.assertEqual(
			checkpoint_manager.checkpoint_folder, './checkpoints',
			f"{checkpoint_manager.checkpoint_folder = }, expected checkpoints"
		)
		self.assertEqual(checkpoint_manager.meta_path_prefix, 'network')
		self.assertEqual(checkpoint_manager.verbose, False)
		self.assertEqual(checkpoint_manager.metric, "val_loss")
		self.assertEqual(checkpoint_manager.minimise_metric, True)
		self.assertEqual(checkpoint_manager.curr_best_metric, np.inf)
	
	@_manage_temp_checkpoints_folder
	def test_constructor_with_args(self):
		"""
		Test that the default constructor works as expected.
		:return: None
		"""
		checkpoint_manager = CheckpointManager(
			checkpoint_folder='./test_checkpoint_folder',
			meta_path_prefix='test_meta_path_prefix',
			verbose=True,
			metric='test_metric',
			minimise_metric=False
		)
		self.assertEqual(
			checkpoint_manager.checkpoint_folder, './test_checkpoint_folder',
			f"{checkpoint_manager.checkpoint_folder = }, expected checkpoints"
		)
		self.assertEqual(checkpoint_manager.meta_path_prefix, 'test_meta_path_prefix')
		self.assertEqual(checkpoint_manager.verbose, True)
		self.assertEqual(checkpoint_manager.metric, "test_metric")
		self.assertEqual(checkpoint_manager.minimise_metric, False)
		self.assertEqual(checkpoint_manager.curr_best_metric, -np.inf)
	
	@_manage_temp_checkpoints_folder
	def test_replace_trainer_history(self):
		trainer = MockTrainer()
		prev_len = len(trainer.callbacks)
		prev_history = trainer.training_history
		new_history = MockHistory(1)
		checkpoint_manager = CheckpointManager("./temp")
		checkpoint_manager._replace_trainer_history(trainer, new_history)
		self.assertEqual(trainer.training_history, new_history)
		self.assertIn(new_history, trainer.callbacks)
		self.assertNotIn(prev_history, trainer.callbacks)
		self.assertEqual(len(trainer.callbacks), prev_len)
		self.assertTrue(trainer.sort_flag)
	
	@_manage_temp_checkpoints_folder
	def test_checkpoints_meta_path_property(self):
		checkpoint_manager = CheckpointManager("./temp")
		self.assertIsInstance(checkpoint_manager.checkpoints_meta_path, str)
		self.assertTrue(checkpoint_manager.checkpoints_meta_path.endswith('.json'))
	
	@_manage_temp_checkpoints_folder
	def test_get_checkpoint_filename(self):
		CheckpointManager.SAVE_EXT = '.pth'
		checkpoint_manager = CheckpointManager("./temp")
		self.assertIsInstance(checkpoint_manager.get_checkpoint_filename(), str)
		self.assertTrue(checkpoint_manager.get_checkpoint_filename().endswith(CheckpointManager.SAVE_EXT))
		for i in range(10):
			self.assertIn(str(i), checkpoint_manager.get_checkpoint_filename(i))
	
	@_manage_temp_checkpoints_folder
	def test_create_new_checkpoint_meta(self):
		checkpoint_manager = CheckpointManager("./temp")
		self.assertIsInstance(checkpoint_manager._create_new_checkpoint_meta(0), dict)
		self.assertIn(
			CheckpointManager.CHECKPOINT_BEST_KEY, checkpoint_manager._create_new_checkpoint_meta(0, best=True)
		)
		self.assertIsInstance(
			checkpoint_manager._create_new_checkpoint_meta(0, best=True)[CheckpointManager.CHECKPOINT_BEST_KEY],
			str
		)
	
	@_manage_temp_checkpoints_folder
	def test_save_checkpoints_meta(self):
		# clear the cache before running the test
		if os.path.exists('./temp'):
			shutil.rmtree("./temp")
		
		# create a new checkpoint manager
		checkpoint_manager = CheckpointManager("./temp")
		
		# check the first checkpoint meta is saved
		test_data = {'test_key': 'test_value'}
		try:
			checkpoint_manager.save_checkpoints_meta(test_data)
			with open(checkpoint_manager.checkpoints_meta_path, "r") as jsonFile:
				info = json.load(jsonFile)
			self.assertEqual(info, test_data)
		except Exception as e:
			self.fail(f"Exception raised: {e}")
		finally:
			if os.path.exists('./temp'):
				shutil.rmtree("./temp")
		
		# check that the updated checkpoint meta is saved
		new_test_data = {'test_key_sec': 'test_value_sec'}
		try:
			checkpoint_manager.save_checkpoints_meta(test_data)
			checkpoint_manager.save_checkpoints_meta(new_test_data)
			with open(checkpoint_manager.checkpoints_meta_path, "r") as jsonFile:
				info = json.load(jsonFile)
			self.assertEqual(info, dict(**test_data, **new_test_data))
		except Exception as e:
			self.fail(f"Exception raised: {e}")
		finally:
			if os.path.exists('./temp'):
				shutil.rmtree("./temp")
	
	@_manage_temp_checkpoints_folder
	def test_save_checkpoint_best(self):
		# create a new checkpoint manager
		checkpoint_manager = CheckpointManager("./temp")
		test_data = {
			CheckpointManager.CHECKPOINT_ITR_KEY                 : 0,
			CheckpointManager.CHECKPOINT_METRICS_KEY             : 'itr_metrics',
			CheckpointManager.CHECKPOINT_STATE_DICT_KEY          : {'test_key': 'test_value'},
			CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: {'test_key_opt': 'test_value_opt'},
			CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY    : 'training_history',
		}
		save_path = checkpoint_manager.save_checkpoint(
			itr=test_data[CheckpointManager.CHECKPOINT_ITR_KEY],
			itr_metrics=test_data[CheckpointManager.CHECKPOINT_METRICS_KEY],
			best=True,
			state_dict=test_data[CheckpointManager.CHECKPOINT_STATE_DICT_KEY],
			optimizer_state_dict=test_data[CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY],
			training_history=test_data[CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY]
		)
		self.assertTrue(save_path.endswith(CheckpointManager.SAVE_EXT))
		saved_data = torch.load(save_path)
		self.assertEqual(saved_data, test_data)
		save_name = checkpoint_manager.get_checkpoint_filename(test_data[CheckpointManager.CHECKPOINT_ITR_KEY])
		with open(checkpoint_manager.checkpoints_meta_path, "r") as jsonFile:
			meta_info = json.load(jsonFile)
		self.assertEqual(meta_info[CheckpointManager.CHECKPOINT_ITRS_KEY][str(0)], save_name)
		self.assertEqual(meta_info[CheckpointManager.CHECKPOINT_BEST_KEY], save_name)
	
	@_manage_temp_checkpoints_folder
	def test_save_checkpoint(self):
		# create a new checkpoint manager
		checkpoint_manager = CheckpointManager("./temp")
		test_data = {
			CheckpointManager.CHECKPOINT_ITR_KEY                 : 0,
			CheckpointManager.CHECKPOINT_METRICS_KEY             : 'itr_metrics',
			CheckpointManager.CHECKPOINT_STATE_DICT_KEY          : {'test_key': 'test_value'},
			CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: {'test_key_opt': 'test_value_opt'},
			CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY    : 'training_history',
		}
		save_path = checkpoint_manager.save_checkpoint(
			itr=test_data[CheckpointManager.CHECKPOINT_ITR_KEY],
			itr_metrics=test_data[CheckpointManager.CHECKPOINT_METRICS_KEY],
			best=False,
			state_dict=test_data[CheckpointManager.CHECKPOINT_STATE_DICT_KEY],
			optimizer_state_dict=test_data[CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY],
			training_history=test_data[CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY]
		)
		self.assertTrue(save_path.endswith(CheckpointManager.SAVE_EXT))
		saved_data = torch.load(save_path)
		self.assertEqual(saved_data, test_data)
		save_name = checkpoint_manager.get_checkpoint_filename(test_data[CheckpointManager.CHECKPOINT_ITR_KEY])
		with open(checkpoint_manager.checkpoints_meta_path, "r") as jsonFile:
			meta_info = json.load(jsonFile)
		self.assertEqual(meta_info[CheckpointManager.CHECKPOINT_ITRS_KEY][str(0)], save_name)
		self.assertNotIn(CheckpointManager.CHECKPOINT_BEST_KEY, meta_info)

	@_manage_temp_checkpoints_folder
	def test_save_checkpoint_multiple_save(self):
		# create a new checkpoint manager
		checkpoint_manager = CheckpointManager("./temp")
		path_list = []
		save_name_list = []
		for i in range(10):
			best = i % 2 == 0
			test_data = {
				CheckpointManager.CHECKPOINT_ITR_KEY                 : i,
				CheckpointManager.CHECKPOINT_METRICS_KEY             : 'itr_metrics',
				CheckpointManager.CHECKPOINT_STATE_DICT_KEY          : {'test_key': 'test_value'},
				CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: {'test_key_opt': 'test_value_opt'},
				CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY    : 'training_history',
			}
			save_path = checkpoint_manager.save_checkpoint(
				itr=test_data[CheckpointManager.CHECKPOINT_ITR_KEY],
				itr_metrics=test_data[CheckpointManager.CHECKPOINT_METRICS_KEY],
				best=best,
				state_dict=test_data[CheckpointManager.CHECKPOINT_STATE_DICT_KEY],
				optimizer_state_dict=test_data[CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY],
				training_history=test_data[CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY]
			)
			self.assertTrue(save_path.endswith(CheckpointManager.SAVE_EXT))
			saved_data = torch.load(save_path)
			self.assertEqual(saved_data, test_data)
			save_name = checkpoint_manager.get_checkpoint_filename(test_data[CheckpointManager.CHECKPOINT_ITR_KEY])
			
			path_list.append(save_path)
			save_name_list.append(save_name)
			self.assertEqual(len(set(path_list)), len(path_list))
			self.assertEqual(len(set(save_name_list)), len(save_name_list))
			
			with open(checkpoint_manager.checkpoints_meta_path, "r") as jsonFile:
				meta_info = json.load(jsonFile)
			self.assertEqual(meta_info[CheckpointManager.CHECKPOINT_ITRS_KEY][str(i)], save_name)
			if best:
				self.assertEqual(meta_info[CheckpointManager.CHECKPOINT_BEST_KEY], save_name)
			else:
				self.assertNotEqual(meta_info.get(CheckpointManager.CHECKPOINT_BEST_KEY), save_name)
			
			for j in range(i):
				self.assertEqual(meta_info[CheckpointManager.CHECKPOINT_ITRS_KEY][str(j)], save_name_list[j])
				self.assertTrue(os.path.exists(path_list[j]))
	
	@_manage_temp_checkpoints_folder
	def test_load_checkpoint_layer_best(self):
		# create model est optimizer
		model = LIFLayer(10, 10)
		model.build()
		opt = torch.optim.Adam(model.parameters(), lr=0.1)
		
		# create a new checkpoint manager
		checkpoint_manager = CheckpointManager("./temp")
		test_data = {
			CheckpointManager.CHECKPOINT_ITR_KEY                 : 0,
			CheckpointManager.CHECKPOINT_METRICS_KEY             : 'itr_metrics',
			CheckpointManager.CHECKPOINT_STATE_DICT_KEY          : model.state_dict(),
			CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: opt.state_dict(),
			CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY    : 'training_history',
		}
		save_path = checkpoint_manager.save_checkpoint(
			itr=test_data[CheckpointManager.CHECKPOINT_ITR_KEY],
			itr_metrics=test_data[CheckpointManager.CHECKPOINT_METRICS_KEY],
			best=True,
			state_dict=test_data[CheckpointManager.CHECKPOINT_STATE_DICT_KEY],
			optimizer_state_dict=test_data[CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY],
			training_history=test_data[CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY]
		)
		
		checkpoint = checkpoint_manager.load_checkpoint(load_checkpoint_mode=LoadCheckpointMode.BEST_ITR)
		self.assertTrue(
			all(
				torch.allclose(v, checkpoint[CheckpointManager.CHECKPOINT_STATE_DICT_KEY][k])
				for k, v in model.state_dict().items()
			)
		)
		self.assertTrue(
			all(
				np.allclose(v, second_param_group[k])
				for first_param_group, second_param_group in zip(
					opt.state_dict()['param_groups'],
					checkpoint[CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY]['param_groups']
				)
				for k, v in first_param_group.items()
			)
		)
	
	@_manage_temp_checkpoints_folder
	def test_load_checkpoint_layer_not_best(self):
		# create model est optimizer
		model = LIFLayer(10, 10)
		model.build()
		opt = torch.optim.Adam(model.parameters(), lr=0.1)
		
		# create a new checkpoint manager
		checkpoint_manager = CheckpointManager("./temp")
		test_data = {
			CheckpointManager.CHECKPOINT_ITR_KEY                 : 0,
			CheckpointManager.CHECKPOINT_METRICS_KEY             : 'itr_metrics',
			CheckpointManager.CHECKPOINT_STATE_DICT_KEY          : model.state_dict(),
			CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: opt.state_dict(),
			CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY    : 'training_history',
		}
		save_path = checkpoint_manager.save_checkpoint(
			itr=test_data[CheckpointManager.CHECKPOINT_ITR_KEY],
			itr_metrics=test_data[CheckpointManager.CHECKPOINT_METRICS_KEY],
			best=False,
			state_dict=test_data[CheckpointManager.CHECKPOINT_STATE_DICT_KEY],
			optimizer_state_dict=test_data[CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY],
			training_history=test_data[CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY]
		)
		
		checkpoint = checkpoint_manager.load_checkpoint(load_checkpoint_mode=LoadCheckpointMode.LAST_ITR)
		self.assertTrue(
			all(
				torch.allclose(v, checkpoint[CheckpointManager.CHECKPOINT_STATE_DICT_KEY][k])
				for k, v in model.state_dict().items()
			)
		)
		self.assertTrue(
			all(
				np.allclose(v, second_param_group[k])
				for first_param_group, second_param_group in zip(
					opt.state_dict()['param_groups'],
					checkpoint[CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY]['param_groups']
				)
				for k, v in first_param_group.items()
			)
		)

	@_manage_temp_checkpoints_folder
	def test_load_checkpoint_sequential_best(self):
		# create model est optimizer
		model = SequentialModel(layers=[LIFLayer(10, 10), LIFLayer(10, 10), LIFLayer(10, 10)])
		model.build()
		opt = torch.optim.Adam(model.parameters(), lr=0.1)
		
		# create a new checkpoint manager
		checkpoint_manager = CheckpointManager("./temp")
		test_data = {
			CheckpointManager.CHECKPOINT_ITR_KEY                 : 0,
			CheckpointManager.CHECKPOINT_METRICS_KEY             : 'itr_metrics',
			CheckpointManager.CHECKPOINT_STATE_DICT_KEY          : model.state_dict(),
			CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: opt.state_dict(),
			CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY    : 'training_history',
		}
		save_path = checkpoint_manager.save_checkpoint(
			itr=test_data[CheckpointManager.CHECKPOINT_ITR_KEY],
			itr_metrics=test_data[CheckpointManager.CHECKPOINT_METRICS_KEY],
			best=True,
			state_dict=test_data[CheckpointManager.CHECKPOINT_STATE_DICT_KEY],
			optimizer_state_dict=test_data[CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY],
			training_history=test_data[CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY]
		)
		
		checkpoint = checkpoint_manager.load_checkpoint(load_checkpoint_mode=LoadCheckpointMode.BEST_ITR)
		self.assertTrue(
			all(
				torch.allclose(v, checkpoint[CheckpointManager.CHECKPOINT_STATE_DICT_KEY][k])
				for k, v in model.state_dict().items()
			)
		)
		self.assertTrue(
			all(
				np.allclose(v, second_param_group[k])
				for first_param_group, second_param_group in zip(
					opt.state_dict()['param_groups'],
					checkpoint[CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY]['param_groups']
				)
				for k, v in first_param_group.items()
			)
		)
	
	@_manage_temp_checkpoints_folder
	def test_load_checkpoint_sequential_not_best(self):
		# create model est optimizer
		model = SequentialModel(layers=[LIFLayer(10, 10), LIFLayer(10, 10), LIFLayer(10, 10)])
		model.build()
		opt = torch.optim.Adam(model.parameters(), lr=0.1)
		
		# create a new checkpoint manager
		checkpoint_manager = CheckpointManager("./temp")
		test_data = {
			CheckpointManager.CHECKPOINT_ITR_KEY                 : 0,
			CheckpointManager.CHECKPOINT_METRICS_KEY             : 'itr_metrics',
			CheckpointManager.CHECKPOINT_STATE_DICT_KEY          : model.state_dict(),
			CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: opt.state_dict(),
			CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY    : 'training_history',
		}
		save_path = checkpoint_manager.save_checkpoint(
			itr=test_data[CheckpointManager.CHECKPOINT_ITR_KEY],
			itr_metrics=test_data[CheckpointManager.CHECKPOINT_METRICS_KEY],
			best=False,
			state_dict=test_data[CheckpointManager.CHECKPOINT_STATE_DICT_KEY],
			optimizer_state_dict=test_data[CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY],
			training_history=test_data[CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY]
		)
		
		checkpoint = checkpoint_manager.load_checkpoint(load_checkpoint_mode=LoadCheckpointMode.LAST_ITR)
		self.assertTrue(
			all(
				torch.allclose(v, checkpoint[CheckpointManager.CHECKPOINT_STATE_DICT_KEY][k])
				for k, v in model.state_dict().items()
			)
		)
		self.assertTrue(
			all(
				np.allclose(v, second_param_group[k])
				for first_param_group, second_param_group in zip(
					opt.state_dict()['param_groups'],
					checkpoint[CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY]['param_groups']
				)
				for k, v in first_param_group.items()
			)
		)
	
	@_manage_temp_checkpoints_folder
	def test_start_overwrite_minimise(self):
		trainer = MockTrainer()
		trainer.load_checkpoint_mode = None
		trainer.force_overwrite = True
		checkpoint_manager = CheckpointManager("./temp", minimise_metric=True)
		checkpoint_manager.save_checkpoints_meta({"test": "test"})
		self.assertTrue(os.path.exists("./temp"))
		checkpoint_manager.start(trainer)
		self.assertFalse(os.path.exists("./temp"))
		self.assertEqual(trainer.current_training_state.iteration, 0)
		self.assertTrue(trainer.training_history.min_call_flag)
		self.assertFalse(trainer.training_history.max_call_flag)
	
	@_manage_temp_checkpoints_folder
	def test_start_overwrite_maximise(self):
		trainer = MockTrainer()
		trainer.load_checkpoint_mode = None
		trainer.force_overwrite = True
		checkpoint_manager = CheckpointManager("./temp", minimise_metric=False)
		checkpoint_manager.save_checkpoints_meta({"test": "test"})
		self.assertTrue(os.path.exists("./temp"))
		checkpoint_manager.start(trainer)
		self.assertFalse(os.path.exists("./temp"))
		self.assertEqual(trainer.current_training_state.iteration, 0)
		self.assertFalse(trainer.training_history.min_call_flag)
		self.assertTrue(trainer.training_history.max_call_flag)
	
	@_manage_temp_checkpoints_folder
	def test_start_load_last_minimise(self):
		# create model est optimizer
		model = SequentialModel(layers=[LIFLayer(10, 10), LIFLayer(10, 10), LIFLayer(10, 10)])
		model.build()
		opt = torch.optim.Adam(model.parameters(), lr=0.1)
		
		# create a new checkpoint manager
		checkpoint_manager = CheckpointManager("./temp", minimise_metric=True)
		new_history = MockHistory(1)
		test_data = {
			CheckpointManager.CHECKPOINT_ITR_KEY                 : 10,
			CheckpointManager.CHECKPOINT_METRICS_KEY             : 'itr_metrics',
			CheckpointManager.CHECKPOINT_STATE_DICT_KEY          : model.state_dict(),
			CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: opt.state_dict(),
			CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY    : new_history,
		}
		
		trainer = MockTrainer()
		trainer.load_checkpoint_mode = LoadCheckpointMode.LAST_ITR
		trainer.model = model
		trainer.optimizer = opt
		trainer.force_overwrite = False
		prev_len = len(trainer.callbacks)
		prev_history = trainer.training_history
		
		save_path = checkpoint_manager.save_checkpoint(
			itr=test_data[CheckpointManager.CHECKPOINT_ITR_KEY],
			itr_metrics=test_data[CheckpointManager.CHECKPOINT_METRICS_KEY],
			best=False,
			state_dict=test_data[CheckpointManager.CHECKPOINT_STATE_DICT_KEY],
			optimizer_state_dict=test_data[CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY],
			training_history=test_data[CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY]
		)
		self.assertTrue(os.path.exists(save_path))
		checkpoint_manager.start(trainer)
		self.assertEqual(
			trainer.current_training_state.iteration,
			test_data[CheckpointManager.CHECKPOINT_ITR_KEY]+1
		)
		self.assertEqual(
			trainer.training_history,
			test_data[CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY]
		)
		self.assertTrue(trainer.training_history.min_call_flag, f"history min call not called.")
		self.assertFalse(trainer.training_history.max_call_flag, f"history max call was called.")
		self.assertIn(new_history, trainer.callbacks)
		self.assertNotIn(prev_history, trainer.callbacks)
		self.assertEqual(len(trainer.callbacks), prev_len)
		self.assertTrue(trainer.sort_flag)
		
		self.assertTrue(
			all(
				torch.allclose(v, trainer.model.state_dict()[k])
				for k, v in model.state_dict().items()
			)
		)
		self.assertTrue(
			all(
				np.allclose(v, second_param_group[k])
				for first_param_group, second_param_group in zip(
					opt.state_dict()['param_groups'],
					trainer.optimizer.state_dict()['param_groups']
				)
				for k, v in first_param_group.items()
			)
		)
	
	@_manage_temp_checkpoints_folder
	def test_start_load_last_maximise(self):
		# create model est optimizer
		model = SequentialModel(layers=[LIFLayer(10, 10), LIFLayer(10, 10), LIFLayer(10, 10)])
		model.build()
		opt = torch.optim.Adam(model.parameters(), lr=0.1)
		
		# create a new checkpoint manager
		checkpoint_manager = CheckpointManager("./temp", minimise_metric=False)
		new_history = MockHistory(1)
		test_data = {
			CheckpointManager.CHECKPOINT_ITR_KEY                 : 10,
			CheckpointManager.CHECKPOINT_METRICS_KEY             : 'itr_metrics',
			CheckpointManager.CHECKPOINT_STATE_DICT_KEY          : model.state_dict(),
			CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: opt.state_dict(),
			CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY    : new_history,
		}
		
		trainer = MockTrainer()
		trainer.load_checkpoint_mode = LoadCheckpointMode.LAST_ITR
		trainer.model = model
		trainer.optimizer = opt
		trainer.force_overwrite = False
		prev_len = len(trainer.callbacks)
		prev_history = trainer.training_history
		
		save_path = checkpoint_manager.save_checkpoint(
			itr=test_data[CheckpointManager.CHECKPOINT_ITR_KEY],
			itr_metrics=test_data[CheckpointManager.CHECKPOINT_METRICS_KEY],
			best=False,
			state_dict=test_data[CheckpointManager.CHECKPOINT_STATE_DICT_KEY],
			optimizer_state_dict=test_data[CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY],
			training_history=test_data[CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY]
		)
		self.assertTrue(os.path.exists(save_path))
		checkpoint_manager.start(trainer)
		self.assertEqual(
			trainer.current_training_state.iteration,
			test_data[CheckpointManager.CHECKPOINT_ITR_KEY] + 1
		)
		self.assertEqual(
			trainer.training_history,
			test_data[CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY]
		)
		self.assertFalse(trainer.training_history.min_call_flag)
		self.assertTrue(trainer.training_history.max_call_flag)
		self.assertIn(new_history, trainer.callbacks)
		self.assertNotIn(prev_history, trainer.callbacks)
		self.assertEqual(len(trainer.callbacks), prev_len)
		self.assertTrue(trainer.sort_flag)
		
		self.assertTrue(
			all(
				torch.allclose(v, trainer.model.state_dict()[k])
				for k, v in model.state_dict().items()
			)
		)
		self.assertTrue(
			all(
				np.allclose(v, second_param_group[k])
				for first_param_group, second_param_group in zip(
					opt.state_dict()['param_groups'],
					trainer.optimizer.state_dict()['param_groups']
				)
				for k, v in first_param_group.items()
			)
		)
	
	@_manage_temp_checkpoints_folder
	def test_start_load_best_minimise(self):
		# create model est optimizer
		model = SequentialModel(layers=[LIFLayer(10, 10), LIFLayer(10, 10), LIFLayer(10, 10)])
		model.build()
		opt = torch.optim.Adam(model.parameters(), lr=0.1)
		
		# create a new checkpoint manager
		checkpoint_manager = CheckpointManager("./temp", minimise_metric=True)
		new_history = MockHistory(1)
		test_data = {
			CheckpointManager.CHECKPOINT_ITR_KEY                 : 10,
			CheckpointManager.CHECKPOINT_METRICS_KEY             : 'itr_metrics',
			CheckpointManager.CHECKPOINT_STATE_DICT_KEY          : model.state_dict(),
			CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: opt.state_dict(),
			CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY    : new_history,
		}
		
		trainer = MockTrainer()
		trainer.load_checkpoint_mode = LoadCheckpointMode.BEST_ITR
		trainer.model = model
		trainer.optimizer = opt
		trainer.force_overwrite = False
		prev_len = len(trainer.callbacks)
		prev_history = trainer.training_history
		
		save_path = checkpoint_manager.save_checkpoint(
			itr=test_data[CheckpointManager.CHECKPOINT_ITR_KEY],
			itr_metrics=test_data[CheckpointManager.CHECKPOINT_METRICS_KEY],
			best=True,
			state_dict=test_data[CheckpointManager.CHECKPOINT_STATE_DICT_KEY],
			optimizer_state_dict=test_data[CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY],
			training_history=test_data[CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY]
		)
		self.assertTrue(os.path.exists(save_path))
		checkpoint_manager.start(trainer)
		self.assertEqual(
			trainer.current_training_state.iteration,
			test_data[CheckpointManager.CHECKPOINT_ITR_KEY] + 1
		)
		self.assertEqual(
			trainer.training_history,
			test_data[CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY]
		)
		self.assertTrue(trainer.training_history.min_call_flag, f"history min call not called.")
		self.assertFalse(trainer.training_history.max_call_flag, f"history max call was called.")
		self.assertIn(new_history, trainer.callbacks)
		self.assertNotIn(prev_history, trainer.callbacks)
		self.assertEqual(len(trainer.callbacks), prev_len)
		self.assertTrue(trainer.sort_flag)
		
		self.assertTrue(
			all(
				torch.allclose(v, trainer.model.state_dict()[k])
				for k, v in model.state_dict().items()
			)
		)
		self.assertTrue(
			all(
				np.allclose(v, second_param_group[k])
				for first_param_group, second_param_group in zip(
					opt.state_dict()['param_groups'],
					trainer.optimizer.state_dict()['param_groups']
				)
				for k, v in first_param_group.items()
			)
		)
		save_name = checkpoint_manager.get_checkpoint_filename(test_data[CheckpointManager.CHECKPOINT_ITR_KEY])
		with open(checkpoint_manager.checkpoints_meta_path, "r") as jsonFile:
			meta_info = json.load(jsonFile)
		self.assertEqual(meta_info[CheckpointManager.CHECKPOINT_BEST_KEY], save_name)
	
	@_manage_temp_checkpoints_folder
	def test_start_load_best_maximise(self):
		# create model est optimizer
		model = SequentialModel(layers=[LIFLayer(10, 10), LIFLayer(10, 10), LIFLayer(10, 10)])
		model.build()
		opt = torch.optim.Adam(model.parameters(), lr=0.1)
		
		# create a new checkpoint manager
		checkpoint_manager = CheckpointManager("./temp", minimise_metric=False)
		new_history = MockHistory(1)
		test_data = {
			CheckpointManager.CHECKPOINT_ITR_KEY                 : 10,
			CheckpointManager.CHECKPOINT_METRICS_KEY             : 'itr_metrics',
			CheckpointManager.CHECKPOINT_STATE_DICT_KEY          : model.state_dict(),
			CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: opt.state_dict(),
			CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY    : new_history,
		}
		
		trainer = MockTrainer()
		trainer.load_checkpoint_mode = LoadCheckpointMode.BEST_ITR
		trainer.model = model
		trainer.optimizer = opt
		trainer.force_overwrite = False
		prev_len = len(trainer.callbacks)
		prev_history = trainer.training_history
		
		save_path = checkpoint_manager.save_checkpoint(
			itr=test_data[CheckpointManager.CHECKPOINT_ITR_KEY],
			itr_metrics=test_data[CheckpointManager.CHECKPOINT_METRICS_KEY],
			best=True,
			state_dict=test_data[CheckpointManager.CHECKPOINT_STATE_DICT_KEY],
			optimizer_state_dict=test_data[CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY],
			training_history=test_data[CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY]
		)
		self.assertTrue(os.path.exists(save_path))
		checkpoint_manager.start(trainer)
		self.assertEqual(
			trainer.current_training_state.iteration,
			test_data[CheckpointManager.CHECKPOINT_ITR_KEY] + 1
		)
		self.assertEqual(
			trainer.training_history,
			test_data[CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY]
		)
		self.assertFalse(trainer.training_history.min_call_flag)
		self.assertTrue(trainer.training_history.max_call_flag)
		self.assertIn(new_history, trainer.callbacks)
		self.assertNotIn(prev_history, trainer.callbacks)
		self.assertEqual(len(trainer.callbacks), prev_len)
		self.assertTrue(trainer.sort_flag)
		
		self.assertTrue(
			all(
				torch.allclose(v, trainer.model.state_dict()[k])
				for k, v in model.state_dict().items()
			)
		)
		self.assertTrue(
			all(
				np.allclose(v, second_param_group[k])
				for first_param_group, second_param_group in zip(
					opt.state_dict()['param_groups'],
					trainer.optimizer.state_dict()['param_groups']
				)
				for k, v in first_param_group.items()
			)
		)
		save_name = checkpoint_manager.get_checkpoint_filename(test_data[CheckpointManager.CHECKPOINT_ITR_KEY])
		with open(checkpoint_manager.checkpoints_meta_path, "r") as jsonFile:
			meta_info = json.load(jsonFile)
		self.assertEqual(meta_info[CheckpointManager.CHECKPOINT_BEST_KEY], save_name)


