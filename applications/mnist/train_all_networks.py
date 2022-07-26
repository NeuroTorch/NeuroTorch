import logging

import torch
from pythonbasictools import DeepLib, log_device_setup, logs_file_setup

from applications.mnist.results_generation import get_training_params_space, train_all_params

if __name__ == '__main__':
	logs_file_setup(__file__)
	torch.cuda.set_per_process_memory_fraction(0.8)
	log_device_setup(deepLib=DeepLib.Pytorch)
	df = train_all_params(
		training_params=get_training_params_space(),
		n_iterations=10,
		batch_size=256,
		data_folder="tr_data_fashion_mnist_001",
		verbose=False,
		rm_data_folder_and_restart_all_training=False,
	)
	logging.info(df)

