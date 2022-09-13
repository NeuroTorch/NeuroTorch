import os.path
import pprint
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
from pythonbasictools import DeepLib, log_device_setup, logs_file_setup

from applications.time_series_forecasting_spiking.dataset import get_dataloader
from applications.time_series_forecasting_spiking.results_generation import train_with_params, visualize_forecasting, \
	try_big_predictions, try_all_chunks_predictions
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
			"dataset_length": -1,
			"n_time_steps": 16,
			"n_encoder_steps": 16,
			"n_units": 256,
			"dt": 1e-3,
			"optimizer": "Adam",
			"learning_rate": 5e-5,
			"min_lr": 5e-7,
			"encoder_type": nt.LIFLayer,
			"predictor_type": nt.SpyLIFLayer,
			"use_recurrent_connection": False,
			"seed": seed,
			"smoothing_sigma": 5,
			"reg": "",
			"hh_init": "inputs",
			"learn_decoder": True,
			"decoder_alpha_as_vec": True,
		},
		n_iterations=2048,
		verbose=True,
		show_training=False,
		force_overwrite=False,
		data_folder="test_checkpoints",
		encoder_data_folder="spikes_autoencoder_checkpoints",
		encoder_iterations=2048,
		batch_size=512,
		save_best_only=True,
	)
	pprint.pprint({k: v for k, v in results.items() if k not in ["predictions", "targets"]}, indent=4)
	# results["history"].plot(show=True)
	# visualize_reconstruction(
	# 	results["auto_encoder_training_output"].dataset.data,
	# 	results["auto_encoder_training_output"].spikes_auto_encoder,
	# 	show=True,
	# )
	# viz_target = VisualiseKMeans(
	# 	results["targets"][0].squeeze(),
	# 	shape=nt.Size(
	# 		[
	# 			Dimension(results["targets"].shape[1], dtype=DimensionProperty.TIME, name="Time Steps"),
	# 			Dimension(results["targets"].shape[-1], dtype=DimensionProperty.NONE, name="Neurons"),
	# 		]
	# 	),
	# )
	# viz_target.heatmap(
	# 	filename=f"{results['checkpoint_folder']}/figures/target_heatmap.png",
	# 	show=False
	# )
	# viz_full_target = VisualiseKMeans(
	# 	results["auto_encoder_training_output"].dataset.data.squeeze(),
	# 	shape=nt.Size(
	# 		[
	# 			Dimension(
	# 				results["auto_encoder_training_output"].dataset.data.shape[1],
	# 				dtype=DimensionProperty.TIME, name="Time Steps"
	# 			),
	# 			Dimension(
	# 				results["auto_encoder_training_output"].dataset.data.shape[-1],
	# 				dtype=DimensionProperty.NONE, name="Neurons"
	# 			),
	# 		]
	# 	),
	# )
	# viz_full_target.heatmap(
	# 	filename=f"{results['checkpoint_folder']}/figures/full_target_heatmap.png",
	# 	show=False
	# )
	# visualize_forecasting(
	# 	results["network"],
	# 	results["auto_encoder_training_output"].spikes_auto_encoder,
	# 	results["dataloader"],
	# 	show=False,
	# )
	try_big_predictions(**results, show=True)
	try_all_chunks_predictions(**results, show=True)
