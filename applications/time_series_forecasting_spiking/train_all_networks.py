import logging

import torch
from pythonbasictools import DeepLib, log_device_setup, logs_file_setup

from applications.time_series_forecasting_spiking.results_generation import get_training_params_space, train_all_params
import neurotorch as nt

if __name__ == '__main__':
	logs_file_setup(__file__, add_stdout=False)
	torch.cuda.set_per_process_memory_fraction(0.9)
	log_device_setup(deepLib=DeepLib.Pytorch)
	df = train_all_params(
		# training_params=get_training_params_space(),
		training_params={
			"dataset_name"            : [
				"timeSeries_2020_12_16_cr3_df.npy"
			],
			"n_time_steps"            : [
				4,
				# 8,
				16,
				# 32,
				64,
				# 128,
				# 256,
				# -1
			],
			"n_encoder_steps"         : [
				# 4,
				8,
				16,
				# 32,
				# 64,
			],
			"n_units"                 : [
				# 32,
				# 128,
				256,
				512,
				# 1024,
			],
			"hidden_units"            : [
				0,
				128,
				256,
				512,
				# 1024,
			],
			"encoder_type"            : [
				# nt.LIFLayer,
				# nt.ALIFLayer,
				nt.SpyLIFLayer,
				# nt.SpyALIFLayer,
			],
			# "predictor_type": [
			# 	nt.LIFLayer,
			# 	nt.ALIFLayer,
			# 	nt.SpyLIFLayer,
			# ],
			"optimizer"               : [
				# "SGD",
				"Adam",
				# "Adamax",
				# "RMSprop",
				# "Adagrad",
				# "Adadelta",
				"AdamW",
			],
			"learning_rate"           : [
				5e-5
			],
			"min_lr"                  : [
				1e-5,
				5e-7,
			],
			"use_recurrent_connection": [
				True,
				False,
			],
			"dt"                      : [
				1e-3,
				# 2e-2,
			],
			"smoothing_sigma"         : [
				# 0,
				5,
				# 10,
			],
			"seed"                    : [
				0,
			],
			"reg"                     : [
				"",
			],
			"hh_init"                 : [
				# "inputs",
				"zeros",
				"random",
			],
			"learn_decoder"           : [
				True,
				False
			],
			"decoder_alpha_as_vec"    : [
				True,
				False
			],
		},
		n_iterations=4096,
		data_folder="predictor_checkpoints_005",
		verbose=False,
		rm_data_folder_and_restart_all_training=False,
		encoder_data_folder="spikes_autoencoder_checkpoints_005",
		encoder_iterations=2048,
		batch_size=512,
		save_best_only=True,
		n_workers=2,
	)
	logging.info(df)
