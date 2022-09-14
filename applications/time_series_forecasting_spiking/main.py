import os.path
import pprint
from copy import deepcopy, copy

import matplotlib.pyplot as plt
import torch
from pythonbasictools import DeepLib, log_device_setup, logs_file_setup

from applications.time_series_forecasting_spiking.dataset import get_dataloader
from applications.time_series_forecasting_spiking.results_generation import train_with_params, visualize_forecasting, \
	try_big_predictions, viz_all_chunks_predictions
from applications.time_series_forecasting_spiking.spikes_auto_encoder_training import visualize_reconstruction
from neurotorch import DimensionProperty, Dimension
from neurotorch.utils import set_seed
import neurotorch as nt
from neurotorch.visualisation.time_series_visualisation import VisualiseKMeans


if __name__ == '__main__':
	logs_file_setup(__file__, add_stdout=False)
	log_device_setup(deepLib=DeepLib.Pytorch)
	torch.cuda.set_per_process_memory_fraction(0.5)
	
	seed = 0
	set_seed(seed)
	
	results = train_with_params(
		{
			"dataset_name": "timeSeries_2020_12_16_cr3_df.npy",
			"dataset_length": 342,
			"dataset_randomize_indexes": True,
			"n_time_steps": 64,
			"n_encoder_steps": 16,
			"n_units": 256,
			"dt": 1e-3,
			"optimizer": "Adam",
			"learning_rate": 5e-5,
			"min_lr": 5e-7,
			"encoder_type": nt.SpyLIFLayer,
			# "predictor_type": nt.SpyLIFLayer,
			"use_recurrent_connection": False,
			"seed": seed,
			"smoothing_sigma": 5,
			"reg": "",
			"hh_init": "random",
			"learn_decoder": False,
			"decoder_alpha_as_vec": True,
		},
		n_iterations=4096,
		verbose=True,
		show_training=False,
		force_overwrite=True,
		data_folder="test_checkpoints",
		encoder_data_folder="test_autoencoder_checkpoints",
		encoder_iterations=2048,
		batch_size=512,
		save_best_only=True,
	)
	
	results_view = copy(results)
	for key in results_view:
		if isinstance(results_view[key], torch.Tensor):
			results_view[key] = results_view[key].shape
	pprint.pprint(results_view, indent=4)
	
	try_big_predictions(**results, show=True)
	viz_all_chunks_predictions(**results, show=True)
