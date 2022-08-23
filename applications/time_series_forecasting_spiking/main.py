import pprint

import matplotlib.pyplot as plt
import torch
from pythonbasictools import DeepLib, log_device_setup, logs_file_setup

from applications.time_series_forecasting_spiking.results_generation import train_with_params, visualize_forecasting
from applications.time_series_forecasting_spiking.spikes_auto_encoder_training import visualize_reconstruction
from neurotorch.utils import set_seed
import neurotorch as nt

if __name__ == '__main__':
	logs_file_setup(__file__, add_stdout=False)
	log_device_setup(deepLib=DeepLib.Pytorch)
	torch.cuda.set_per_process_memory_fraction(0.5)
	
	seed = 0
	set_seed(seed)
	
	results = train_with_params(
		{
			"dataset_name": "timeSeries_2020_12_16_cr3_df.npy",
			"n_time_steps": 16,
			"n_encoder_steps": 32,
			"n_units": 128,
			"dt": 1e-3,
			"optimizer": "Adam",
			"learning_rate": 5e-5,
			"min_lr": 5e-6,
			"encoder_type": nt.LIFLayer,
			"use_recurrent_connection": True,
			"seed": seed,
			"smoothing_sigma": 5,
		},
		n_iterations=4096,
		verbose=True,
		show_training=False,
		force_overwrite=True,
		data_folder="predictor_checkpoints",
		encoder_data_folder="autoencoder_checkpoints",
	)
	pprint.pprint(results, indent=4)
	results["history"].plot(show=True)
	visualize_reconstruction(
		results["auto_encoder_training_output"].dataset.data,
		results["auto_encoder_training_output"].spikes_auto_encoder,
		show=True,
	)
	visualize_forecasting(
		results["network"],
		results["auto_encoder_training_output"].spikes_auto_encoder,
		results["dataloader"],
		show=True,
	)
