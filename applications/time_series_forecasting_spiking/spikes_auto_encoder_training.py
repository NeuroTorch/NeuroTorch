import logging
import os
import shutil
import warnings
from typing import NamedTuple, Type, Union, Tuple, Any, Dict, Optional

import pandas as pd
import psutil
import tracemalloc
import torchvision
from matplotlib import pyplot as plt
import tqdm
from pythonbasictools import logs_file_setup, log_device_setup, DeepLib

import neurotorch as nt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage import gaussian_filter1d

from applications.util import get_optimizer
from neurotorch import to_tensor
from neurotorch.transforms.spikes_auto_encoder import SpikesAutoEncoder
from neurotorch.transforms.spikes_encoders import SpikesEncoder
from neurotorch.utils import hash_params, set_seed, save_params, get_all_params_combinations


class TimeSeriesAutoEncoderDataset(Dataset):
	def __init__(
			self,
			n_units: int = 128,
			seed: int = 0,
			filename: Optional[str] = None,
			smoothing_sigma: float = 0.0,
			**kwargs
	):
		super().__init__()
		if filename is None:
			filename = 'timeSeries_2020_12_16_cr3_df.npy'
		self.random_state = np.random.RandomState(seed)
		self.ts = np.load(filename)
		self.n_neurons, self.n_time_steps = self.ts.shape
		if n_units <= 0:
			n_units = self.n_neurons
		self.n_units = n_units
		self.units_indexes = self.random_state.randint(self.n_neurons, size=self.n_units)
		self.data = self.ts[self.units_indexes, :]
		self.sigma = smoothing_sigma
		
		for neuron in range(self.data.shape[0]):
			if self.sigma > 0.0:
				self.data[neuron, :] = gaussian_filter1d(self.data[neuron, :], sigma=self.sigma)
			self.data[neuron, :] = self.data[neuron, :] - np.min(self.data[neuron, :])
			self.data[neuron, :] = self.data[neuron, :] / np.max(self.data[neuron, :])
		
		self.data = nt.to_tensor(self.data.T, dtype=torch.float32)
	
	def __len__(self):
		return self.n_time_steps
	
	def __getitem__(self, item):
		return torch.unsqueeze(self.data[item], dim=0), torch.unsqueeze(self.data[item], dim=0)


def compute_reconstruction_pvar(
		time_series,
		auto_encoder: SpikesAutoEncoder,
):
	target = nt.to_tensor(time_series, dtype=torch.float32)
	
	with torch.no_grad():
		spikes = auto_encoder.encode(torch.unsqueeze(target, dim=1))
		out = auto_encoder.decode(spikes)
		spikes = torch.squeeze(spikes).detach().cpu().numpy()
		predictions = torch.squeeze(out.detach().cpu())
		target = torch.squeeze(target.detach().cpu())
		
		errors = torch.squeeze(predictions - target.to(predictions.device))**2
		mse_loss = torch.nn.MSELoss()(predictions, target.to(predictions.device))
		pVar = 1 - mse_loss / torch.var(target.to(mse_loss.device))
	return pVar.detach().cpu().item()


def visualize_reconstruction(
		time_series,
		auto_encoder: SpikesAutoEncoder,
		filename: Optional[str] = None,
		show: bool = False,
) -> plt.Figure:
	target = nt.to_tensor(time_series, dtype=torch.float32)
	
	spikes = auto_encoder.encode(torch.unsqueeze(target, dim=1))
	out = auto_encoder.decode(spikes)
	spikes = torch.squeeze(spikes).detach().cpu().numpy()
	predictions = torch.squeeze(out.detach().cpu())
	target = torch.squeeze(target.detach().cpu())
	
	errors = torch.squeeze(predictions - target.to(predictions.device))**2
	mse_loss = torch.nn.MSELoss()(predictions, target.to(predictions.device))
	pVar = 1 - mse_loss / torch.var(target.to(mse_loss.device))
	
	fig, axes = plt.subplots(3, 1, figsize=(15, 8))
	axes[0].plot(errors.detach().cpu().numpy())
	axes[0].set_xlabel("Time [-]")
	axes[0].set_ylabel("Squared Error [-]")
	axes[0].set_title(
		f"Encoder: {auto_encoder.encoder_type.__name__}<{auto_encoder.n_units}u, {auto_encoder.n_encoder_steps}t>, "
		f"pVar: {pVar.detach().cpu().item():.3f}, "
	)
	
	mean_errors = torch.mean(errors, dim=0)
	mean_error_sort, indices = torch.sort(mean_errors)
	predictions = torch.squeeze(predictions).numpy().T
	target = torch.squeeze(target).numpy().T
	
	best_idx, worst_idx = indices[0], indices[-1]
	show_single_preds(
		auto_encoder, axes[1], predictions[best_idx], target[best_idx], spikes[:, :, best_idx],
		title="Best reconstruction"
	)
	show_single_preds(
		auto_encoder, axes[2], predictions[worst_idx], target[worst_idx], spikes[:, :, worst_idx],
		title="Worst reconstruction"
	)
	
	fig.set_tight_layout(True)
	if filename is not None:
		os.makedirs(os.path.dirname(filename), exist_ok=True)
		fig.savefig(filename)
	if show:
		plt.show()
	plt.close(fig)
	return fig


def show_single_preds(auto_encoder, ax, predictions, target, spikes, title=""):
	predictions, target = to_tensor(predictions), to_tensor(target)
	mse_loss = torch.nn.MSELoss()(predictions, target.to(predictions.device))
	pVar = 1 - mse_loss / torch.var(target.to(mse_loss.device))
	y_max = max(target.max(), predictions.max())
	x_scatter_space = np.linspace(0, len(target), num=auto_encoder.n_encoder_steps * len(target))
	x_scatter_spikes = []
	x_scatter_zeros = []
	for i, xs in enumerate(x_scatter_space):
		if np.isclose(
				spikes[i // auto_encoder.n_encoder_steps][i % auto_encoder.n_encoder_steps],
				1.0
		):
			x_scatter_spikes.append(xs)
		else:
			x_scatter_zeros.append(xs)
	ax.plot(predictions.detach().cpu().numpy(), label=f"Prediction (pVar: {pVar.detach().cpu().item():.3f})")
	ax.plot(target.detach().cpu().numpy(), label="Target")
	ax.scatter(
		x_scatter_spikes, y=[y_max * 1.1] * len(x_scatter_spikes),
		label="Latent space", c='k', marker='|', linewidths=0.5
	)
	ax.set_xlabel("Time [-]")
	ax.set_ylabel("Activity [-]")
	ax.set_title(title)
	ax.legend()


class AutoEncoderTrainingOutput(NamedTuple):
	spikes_auto_encoder: SpikesAutoEncoder
	history: nt.TrainingHistory
	dataset: TimeSeriesAutoEncoderDataset
	checkpoints_name: str
	params: dict
	train_loss: float
	pVar: float


def train_auto_encoder(
		encoder_type: Type[Union[nt.LIFLayer, nt.SpyLIFLayer, nt.ALIFLayer]],
		n_units: int,
		n_encoder_steps: int,
		dt: float = 1e-3,
		batch_size: int = 256,
		n_iterations: int = 4096,
		seed: int = 0,
		load_and_save: bool = True,
		data_folder: Optional[str] = "spikes_autoencoder_checkpoints",
		verbose: bool = True,
		force_overwrite: bool = False,
		**kwargs
) -> AutoEncoderTrainingOutput:
	kwargs.setdefault("use_recurrent_connection", True)
	kwargs.setdefault("optimizer", "AdamW")
	kwargs.setdefault("smoothing_sigma", 0.0)
	set_seed(seed)
	dataset = TimeSeriesAutoEncoderDataset(n_units=n_units, seed=seed, filename=kwargs.get("dataset_name"), **kwargs)
	n_units = dataset.n_units
	params = dict(
		encoder_type=encoder_type,
		n_units=n_units,
		n_encoder_steps=n_encoder_steps,
		dt=dt,
		batch_size=batch_size,
		n_iterations=n_iterations,
		seed=seed,
		optimizer=kwargs.get("optimizer"),
		use_recurrent_connection=kwargs.get("use_recurrent_connection"),
		dataset_name=kwargs.get("dataset_name"),
		smoothing_sigma=kwargs.get("smoothing_sigma"),
	)
	checkpoints_name = str(hash_params(params))
	checkpoint_folder = f"{data_folder}/{checkpoints_name}"
	os.makedirs(checkpoint_folder, exist_ok=True)
	save_params(params, os.path.join(checkpoint_folder, "params.pkl"))
	checkpoint_manager = nt.CheckpointManager(
		checkpoint_folder, metric="train_loss", minimise_metric=True, save_freq=-1
	)
	dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
	spikes_auto_encoder = SpikesAutoEncoder(
		n_units, n_encoder_steps=n_encoder_steps, encoder_type=encoder_type, checkpoint_folder=checkpoint_folder, dt=dt,
		**kwargs
	).build()
	if verbose:
		logging.info(f"\nNetwork:\n{spikes_auto_encoder}")
	trainer = nt.Trainer(
		spikes_auto_encoder,
		callbacks=([checkpoint_manager] if load_and_save else None),
		optimizer=get_optimizer(kwargs["optimizer"])(
			spikes_auto_encoder.parameters(), lr=2e-4, weight_decay=0.0
		),
		verbose=verbose,
	)
	history = trainer.train(
		dataloader,
		n_iterations=n_iterations,
		load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR if not force_overwrite else None,
		force_overwrite=force_overwrite,
		desc=f"Training {checkpoints_name}:{encoder_type.__name__}<{n_units}u, {n_encoder_steps}t>",
	)
	visualize_reconstruction(
		dataset.data, spikes_auto_encoder,
		filename=f"{checkpoint_folder}/figures/reconstruction_visualization.png",
		show=False,
	)
	return AutoEncoderTrainingOutput(
		spikes_auto_encoder=spikes_auto_encoder,
		history=history,
		dataset=dataset,
		checkpoints_name=checkpoints_name,
		params=params,
		train_loss=history["train_loss"][-1],
		pVar=compute_reconstruction_pvar(dataset.data, spikes_auto_encoder),
	)


def display_top(snapshot: tracemalloc.Snapshot, key_type='lineno', limit=3):
	import linecache
	import os
	
	snapshot = snapshot.filter_traces(
		(
			tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
			tracemalloc.Filter(False, "<unknown>"),
		)
	)
	top_stats = snapshot.statistics(key_type)
	
	print(f"Top {limit} lines")
	for index, stat in enumerate(top_stats[:limit], 1):
		frame = stat.traceback[0]
		filename = os.sep.join(frame.filename.split(os.sep)[-2:])
		print(
			f"#{index:2d} {stat.size / 1024:>7.1f} KiB ({stat.size * 1e-9:.3f} GB) \n"
			f"\t{filename}:{frame.lineno} "
		)
		line = linecache.getline(frame.filename, frame.lineno).strip()
		if line:
			print(f'\t> {line}')
	
	other = top_stats[limit:]
	if other:
		size = sum(stat.size for stat in other)
		print(f"{len(other)} other: {size / 1024:.1f} KiB ({size * 1e-9:.3f} GB)")
	total = sum(stat.size for stat in top_stats)
	print(f"Total allocated size: {total / 1024:.1f} KiB ({total * 1e-9:.3f} GB)")

	
def get_training_params_space() -> Dict[str, Any]:
	"""
	Get the parameters space for the training.
	
	:return: The parameters space.
	"""
	return {
		"dataset_name": [
			"timeSeries_2020_12_16_cr3_df.npy"
		],
		"n_encoder_steps": [
			8,
			16,
			32,
			64,
		],
		"n_units": [
			32,
			128,
			1024,
		],
		"encoder_type": [
			nt.LIFLayer,
			nt.ALIFLayer,
			nt.SpyLIFLayer,
		],
		"use_recurrent_connection": [
			True,
			False,
		],
		"dt": [
			1e-3,
			# 2e-2
		],
		"optimizer": [
			# "SGD",
			"Adam",
			# "Adamax",
			# "RMSprop",
			# "Adagrad",
			# "Adadelta",
			"AdamW",
		],
		"smoothing_sigma": [5],
		"seed": [
			0,
		],
	}


def train_all_params(
		training_params: Dict[str, Any] = None,
		n_iterations: int = 4096,
		batch_size: int = 256,
		data_folder: str = "spikes_autoencoder_checkpoints",
		verbose: bool = False,
		rm_data_folder_and_restart_all_training: bool = False,
		force_overwrite: bool = False,
		skip_if_exists: bool = False,
		device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
):
	"""
	Train the network with all the parameters.

	:param n_iterations: The number of iterations to train the network.
	:param batch_size: The batch size to use.
	:param verbose: If True, print the progress of each training.
	:param data_folder: The folder where to save the data.
	:param training_params: The parameters to use for the training.
	:param rm_data_folder_and_restart_all_training: If True, remove the data folder and restart all the training.
	:param force_overwrite: If True, overwrite and restart non-completed training.
	:param skip_if_exists: If True, skip the training if the results already in the results dataframe.
	:return: The results of the training.
	"""
	warnings.filterwarnings("ignore", category=UserWarning)
	if rm_data_folder_and_restart_all_training and os.path.exists(data_folder):
		shutil.rmtree(data_folder)
	os.makedirs(data_folder, exist_ok=True)
	results_path = os.path.join(data_folder, "results.csv")
	if training_params is None:
		training_params = get_training_params_space()
	
	if len(training_params) == 1:
		verbose = True
	
	all_params_combinaison_dict = get_all_params_combinations(training_params)
	columns = [
		'checkpoints',
		*list(training_params.keys()),
		'train_loss', 'pVar',
	]
	
	# load dataframe if exists
	try:
		df = pd.read_csv(results_path)
	except FileNotFoundError:
		df = pd.DataFrame(columns=columns)
	
	with tqdm.tqdm(
			all_params_combinaison_dict, desc="Training all the parameters", position=0, disable=verbose
	) as p_bar:
		for i, params in enumerate(p_bar):
			with torch.no_grad():
				torch.cuda.empty_cache()
			if str(hash_params(params)) in df["checkpoints"].values and skip_if_exists:
				continue
			# p_bar.set_description(f"Training {params}")
			try:
				result = train_auto_encoder(
					**params,
					n_iterations=n_iterations,
					batch_size=batch_size,
					data_folder=data_folder,
					verbose=verbose,
					force_overwrite=force_overwrite,
					device=device,
				)
				params.update(result.params)
				if result.checkpoints_name in df["checkpoints"].values:
					# remove from df if already exists
					df = df[df["checkpoints"] != result.checkpoints_name]
				df = pd.concat(
					[df, pd.DataFrame(
						dict(
							checkpoints=[result.checkpoints_name],
							**{k: [v] for k, v in params.items()},
							train_loss=[result.train_loss],
							pVar=[result.pVar],
						)
					)], ignore_index=True,
				)
				df.to_csv(results_path, index=False)
				p_bar.set_postfix(
					params=params,
					pVar=result.pVar,
				)
			# except Exception as e:
			# 	logging.error(e)
			# 	continue
			finally:
				with torch.no_grad():
					torch.cuda.empty_cache()
	return df


if __name__ == '__main__':
	logs_file_setup(__file__, add_stdout=False)
	torch.cuda.set_per_process_memory_fraction(0.5)
	log_device_setup(deepLib=DeepLib.Pytorch)
	df_results = train_all_params(
		n_iterations=1024,
		training_params=get_training_params_space(),
		verbose=False,
		rm_data_folder_and_restart_all_training=False,
		data_folder="spikes_autoencoder_checkpoints_002",
	)
	logging.info(df_results)
	
	# tracemalloc.start()
	# out = train_auto_encoder(
	# 	nt.SpyLIFLayer, 128, 32,
	# 	n_iterations=4096,
	# 	batch_size=256,
	# 	data_folder="autoencoder_checkpoints",
	# 	verbose=True,
	# 	force_overwrite=True,
	# 	# device=torch.device("cuda:0"),
	# 	smoothing_sigma=5.0,
	# 	**{
	# 		# "hard_tanh.min_val": -1.0,
	# 		"use_recurrent_connection": True,
	# 		# "forward_weights": torch.normal(0.0, 0.2/np.sqrt(128), (128, 128)),
	# 	},
	# )
	# snapshot = tracemalloc.take_snapshot()
	# tracemalloc.stop()
	# display_top(snapshot, limit=5)
