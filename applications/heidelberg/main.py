import pprint

import torch
from pythonbasictools import DeepLib, log_device_setup, logs_file_setup

from applications.heidelberg.results_generation import train_with_params
from neurotorch.modules.layers import LayerType

if __name__ == '__main__':
	logs_file_setup(__file__)
	log_device_setup(deepLib=DeepLib.Pytorch)
	torch.autograd.set_detect_anomaly(True)
	torch.cuda.set_per_process_memory_fraction(0.1)
	results = train_with_params(
		{
			"use_recurrent_connection": True,
			'n_hidden_neurons': 256,
			"n_steps": 100,
			"optimizer": "Adam",
			"learning_rate": 2e-4,
			"hidden_layer_type": LayerType.SpyLIF,
			"readout_layer_type": LayerType.SpyLI,
		},
		n_iterations=50,
		batch_size=256,
		verbose=True,
		show_training=False,
		force_overwrite=True,
		data_folder="tr_test",
	)
	pprint.pprint(results, indent=4)
	results["history"].plot(show=True)
