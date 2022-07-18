import logging

import torch
from pythonbasictools import DeepLib, log_device_setup, logs_file_setup

from applications.mnist.results_generation import get_training_params_space, train_all_params

if __name__ == '__main__':
	logs_file_setup(__file__)
	torch.cuda.set_per_process_memory_fraction(0.9)
	log_device_setup(deepLib=DeepLib.Pytorch)
	df = train_all_params(
		training_params=get_training_params_space(),
		n_iterations=30,
		batch_size=4096,
		data_folder="tr_data",
		verbose=False,
	)
	logging.info(df)

