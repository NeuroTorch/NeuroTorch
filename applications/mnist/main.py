import pprint

import torch
from pythonbasictools.device import DeepLib, log_device_setup
from pythonbasictools.logging import logs_file_setup

from applications.mnist.dataset import DatasetId
from applications.mnist.results_generation import train_with_params
from neurotorch.modules.layers import LayerType, LearningType

if __name__ == '__main__':
	logs_file_setup(__file__)
	log_device_setup(deepLib=DeepLib.Pytorch)
	torch.autograd.set_detect_anomaly(True)
	torch.cuda.set_per_process_memory_fraction(0.5)
	results = train_with_params(
		{
			"dataset_id": DatasetId.MNIST,
			"use_recurrent_connection": False,
			"input_transform": "const",
			'n_hidden_neurons': 128,
			"n_steps": 32,
			"train_val_split_ratio": 0.95,
			# "spike_func": SpikeFuncType.FastSigmoid,
			"input_layer_type": LayerType.SpyLIF,
			"input_learning_type": LearningType.BPTT,
			"hidden_layer_type": LayerType.SpyLIF,
			"readout_layer_type": LayerType.SpyLI,
			"foresight_time_steps": 0,
		},
		n_iterations=30,
		batch_size=128,
		verbose=True,
		show_training=False,
		force_overwrite=False,
		data_folder="tr_test",
	)
	pprint.pprint(results, indent=4)
