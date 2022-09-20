import logging

import torch
from pythonbasictools import DeepLib, log_device_setup, logs_file_setup

from applications.heidelberg.results_generation import get_training_params_space, train_all_params
from neurotorch import LayerType

opt_params_space = {
		"n_steps": [
			100,
		],
		"n_hidden_neurons": [
			256,
		],
		"hidden_layer_type": [
			LayerType.SpyLIF,
		],
		"readout_layer_type": [
			LayerType.SpyLI,
		],
		"use_recurrent_connection": [
			True
		],
		"optimizer": [
			"SGD",
			"Adam",
			"Adamax",
			# "RMSprop",
			# "Adagrad",
			# "Adadelta",
			"AdamW",
		],
		"learning_rate": [
			2e-4,
		],
	}


if __name__ == '__main__':
	logs_file_setup(__file__)
	torch.cuda.set_per_process_memory_fraction(0.5)
	log_device_setup(deepLib=DeepLib.Pytorch)
	df = train_all_params(
		# training_params=get_training_params_space(),
		training_params=opt_params_space,
		n_iterations=50,
		batch_size=256,
		data_folder="tr_data_heidelberg_optimizers_001",
		verbose=False,
		rm_data_folder_and_restart_all_training=False,
	)
	logging.info(df)

