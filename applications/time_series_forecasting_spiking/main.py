import os.path
import pprint
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
from pythonbasictools import DeepLib, log_device_setup, logs_file_setup

from applications.time_series_forecasting_spiking.dataset import get_dataloader
from applications.time_series_forecasting_spiking.results_generation import train_with_params, visualize_forecasting
from applications.time_series_forecasting_spiking.spikes_auto_encoder_training import visualize_reconstruction
from neurotorch import DimensionProperty, Dimension
from neurotorch.utils import set_seed
import neurotorch as nt
from neurotorch.visualisation.time_series_visualisation import VisualiseKMeans


@torch.no_grad()
def try_big_predictions(**kwargs):
	checkpoint_folder = kwargs["checkpoint_folder"]
	spikes_auto_encoder = kwargs["auto_encoder_training_output"].spikes_auto_encoder
	loader_params = deepcopy(kwargs["params"])
	loader_params['n_time_steps'] = -1
	loader_params['dataset_length'] = 1
	dataloader = get_dataloader(
		units=kwargs["auto_encoder_training_output"].dataset.units_indexes, **loader_params
	)
	t0, target = next(iter(dataloader))
	preds, hh = kwargs["network"].get_prediction_trace(
		t0, foresight_time_steps=(target.shape[1] - 1) * kwargs["params"]["n_encoder_steps"], return_hidden_states=True
	)
	spikes_preds = hh[kwargs["network"].get_layer().name][-1]
	spikes = torch.squeeze(spikes_preds).detach().cpu().numpy().reshape(
		-1, spikes_auto_encoder.n_encoder_steps, spikes_auto_encoder.n_units
	)
	viz = VisualiseKMeans(
		preds.detach().cpu().numpy().squeeze(),
		shape=nt.Size(
			[
				Dimension(preds.shape[1], dtype=DimensionProperty.TIME, name="Time Steps"),
				Dimension(preds.shape[-1], dtype=DimensionProperty.NONE, name="Neurons"),
			]
		),
	)
	viz.plot_timeseries_comparison(
		target, spikes,
		n_spikes_steps=results["params"]["n_encoder_steps"],
		title=f"Predictor: {spikes_auto_encoder.encoder_type.__name__}"
		f"<{spikes_auto_encoder.n_units}u, {spikes_auto_encoder.n_encoder_steps}t>",
		desc="Prediction",
		filename=f"{checkpoint_folder}/figures/full_forecasting_visualization.png",
		show=False,
	)
	

@torch.no_grad()
def try_all_small_predictions(**kwargs):
	checkpoint_folder = kwargs["checkpoint_folder"]
	spikes_auto_encoder = kwargs["auto_encoder_training_output"].spikes_auto_encoder
	loader_params = deepcopy(kwargs["params"])
	loader_params['n_time_steps'] = -1
	loader_params['dataset_length'] = 1
	dataloader = get_dataloader(
		units=kwargs["auto_encoder_training_output"].dataset.units_indexes, **loader_params
	)
	n_time_steps = kwargs["params"]["n_time_steps"]
	n_encoder_steps = kwargs["params"]["n_encoder_steps"]
	t0, target = next(iter(dataloader))
	n_complete_chunk = int(target.shape[1] / n_time_steps)
	target = target[:, :n_complete_chunk * n_time_steps]
	target_chunks = target.reshape(
		n_complete_chunk, n_time_steps, target.shape[-1]
	)
	
	preds, hh = kwargs["network"].get_prediction_trace(
		target_chunks[:, None, 0], foresight_time_steps=(n_time_steps-1)*n_encoder_steps, return_hidden_states=True
	)
	preds = preds.reshape(1, -1, preds.shape[-1])
	spikes_preds = hh[kwargs["network"].get_layer().name][-1]
	spikes = torch.squeeze(spikes_preds).detach().cpu().numpy().reshape(
		-1, spikes_auto_encoder.n_encoder_steps, spikes_auto_encoder.n_units
	)
	
	viz = VisualiseKMeans(
		preds.detach().cpu().numpy().squeeze(),
		shape=nt.Size(
			[
				Dimension(preds.shape[1], dtype=DimensionProperty.TIME, name="Time Steps"),
				Dimension(preds.shape[-1], dtype=DimensionProperty.NONE, name="Neurons"),
			]
		),
	)
	filename = f"{checkpoint_folder}/figures/full_chunks_forecasting_visualization.png"
	fig, axes = viz.plot_timeseries_comparison(
		target, spikes,
		n_spikes_steps=results["params"]["n_encoder_steps"],
		title=f"Predictor: {spikes_auto_encoder.encoder_type.__name__}"
		f"<{spikes_auto_encoder.n_units}u, {spikes_auto_encoder.n_encoder_steps}t>",
		desc="Prediction",
		filename=None,
		show=False,
		close=False,
	)
	for i in range(1, n_complete_chunk):
		for ax in axes[1:]:
			ax.vlines(
				i * n_time_steps,
				ymin=torch.min(target).cpu(), ymax=torch.max(target).cpu(),
				color="red", linestyle="-", linewidth=0.5, alpha=0.5,
			)
	os.makedirs(os.path.dirname(filename), exist_ok=True)
	fig.savefig(filename)
	plt.show()
	plt.close(fig)


if __name__ == '__main__':
	logs_file_setup(__file__, add_stdout=False)
	log_device_setup(deepLib=DeepLib.Pytorch)
	torch.cuda.set_per_process_memory_fraction(0.8)
	
	seed = 0
	set_seed(seed)
	
	results = train_with_params(
		{
			"dataset_name": "timeSeries_2020_12_16_cr3_df.npy",
			"dataset_length": -1,
			"n_time_steps": 32,
			"n_encoder_steps": 16,
			"n_units": 256,
			"dt": 1e-3,
			"optimizer": "Adam",
			"learning_rate": 5e-5,
			"min_lr": 5e-7,
			"encoder_type": nt.SpyLIFLayer,
			"predictor_type": nt.SpyLIFLayer,
			"use_recurrent_connection": False,
			"seed": seed,
			"smoothing_sigma": 5,
			"reg": "",
			"hh_init": "inputs",
			"learn_decoder": True,
			"decoder_alpha_as_vec": True,
		},
		n_iterations=1024,
		verbose=True,
		show_training=False,
		force_overwrite=True,
		data_folder="test_checkpoints",
		encoder_data_folder="spikes_autoencoder_checkpoints",
		encoder_iterations=1024,
		batch_size=256,
		save_best_only=True,
	)
	pprint.pprint(results, indent=4)
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
	# try_big_predictions(**results)
	try_all_small_predictions(**results)
