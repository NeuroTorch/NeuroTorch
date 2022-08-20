import pprint

import torch
from pythonbasictools import DeepLib, log_device_setup, logs_file_setup

from applications.time_series_forecasting_spiking.results_generation import train_with_params
from neurotorch.utils import set_seed
import neurotorch as nt

if __name__ == '__main__':
	logs_file_setup(__file__, add_stdout=False)
	log_device_setup(deepLib=DeepLib.Pytorch)
	torch.autograd.set_detect_anomaly(True)
	torch.cuda.set_per_process_memory_fraction(0.5)
	
	seed = 0
	set_seed(seed)
	
	results = train_with_params(
		{
			"dataset_name": "timeSeries_2020_12_16_cr3_df.npy",
			"n_time_steps": 16,
			"n_encoder_steps": 32,
			"n_units": 128,
			"dt": 2e-2,
			"optimizer": "Adam",
			"learning_rate": 5e-5,
			"encoder_type": nt.SpyLIFLayer,
			"use_recurrent_connection": True,
			"seed": seed,
		},
		n_iterations=1024,
		verbose=True,
		show_training=False,
		force_overwrite=True,
		data_folder="tr_results",
	)
	pprint.pprint(results, indent=4)
	results["history"].plot(show=True)
