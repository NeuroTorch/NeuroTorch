from typing import NamedTuple, Type

import psutil
import torchvision
from matplotlib import pyplot as plt

import neurotorch as nt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage import gaussian_filter1d


class TimeSeriesAutoEncoderDataset(Dataset):
	def __init__(self, sample_size=128, seed: int = 0):
		super().__init__()
		self.random_state = np.random.RandomState(seed)
		self.ts = np.load('timeSeries_2020_12_16_cr3_df.npy')
		self.n_neurons, self.n_time_steps = self.ts.shape
		self.sample_size = sample_size
		self.sample_indexes = self.random_state.randint(self.n_neurons, size=self.sample_size)
		self.data = self.ts[self.sample_indexes, :]
		self.sigma = 30
		
		for neuron in range(self.data.shape[0]):
			self.data[neuron, :] = gaussian_filter1d(self.data[neuron, :], sigma=self.sigma)
			self.data[neuron, :] = self.data[neuron, :] - np.min(self.data[neuron, :])
			self.data[neuron, :] = self.data[neuron, :] / np.max(self.data[neuron, :])
		
		self.data = nt.to_tensor(self.data.T, dtype=torch.float32)

	def __len__(self):
		return self.n_time_steps
	
	def __getitem__(self, item):
		return torch.unsqueeze(self.data[item], dim=0), torch.unsqueeze(self.data[item], dim=0)
	

class SpikesAutoEncoder(nt.SequentialModel):
	def __new__(cls, *args, **kwargs):
		return object.__new__(cls)
	
	def __init__(
			self,
			encoder_type: Type[nt.modules.BaseLayer],
			n_units: int,
			n_encoder_steps: int,
			*args,
			**kwargs
	):
		self.encoder_type = encoder_type
		self.n_units = n_units
		self.n_encoder_steps = n_encoder_steps
		spikes_encoder = encoder_type(
			n_units, n_units,
			forward_weights=np.random.random((n_units, n_units)) * 1.5,
			learning_type=nt.LearningType.BPTT,
			use_recurrent_connection=False,
			name='encoder'
		).build()
		conv = torch.nn.Conv1d(
			n_units, n_units, n_encoder_steps,
			stride=n_encoder_steps,
			bias=False
		)
		torch.nn.init.constant_(conv.weight, 1.0/n_encoder_steps)
		# torch.nn.init.constant_(conv.weight, 1.0)
		conv.requires_grad_(False)
		spikes_decoder = torch.nn.Sequential(
			torchvision.ops.Permute([0, 2, 1]),
			conv,
			torchvision.ops.Permute([0, 2, 1]),
		)
		super(SpikesAutoEncoder, self).__init__(
			input_transform=[
				nt.transforms.ConstantValuesTransform(
					n_steps=n_encoder_steps,
				)
			],
			layers=[spikes_encoder],
			output_transform=[spikes_decoder],
			*args, **kwargs
		)
		self.spikes_encoder = spikes_encoder
		self.spikes_decoder = spikes_decoder
	
	def encode(self, x: torch.Tensor) -> torch.Tensor:
		x = self._inputs_to_dict(x)
		x = self.apply_input_transform(x)
		x = x[self.spikes_encoder.name]
		out, hh = [], None
		for t in range(x.shape[1]):
			out_t, hh = self.spikes_encoder(x[:, t], hh)
			out.append(out_t)
		return torch.stack(out, dim=1)
	
	def decode(self, x: torch.Tensor):
		return self.spikes_decoder(x)
	

def show_prediction(time_series, auto_encoder: SpikesAutoEncoder):
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
	_show_single_preds(
		auto_encoder, axes[1], predictions[best_idx], target[best_idx], spikes[:, :, best_idx],
		title="Best reconstruction"
	)
	_show_single_preds(
		auto_encoder, axes[2], predictions[worst_idx], target[worst_idx], spikes[:, :, worst_idx],
		title="Worst reconstruction"
	)
	
	fig.set_tight_layout(True)
	plt.show()


def _show_single_preds(auto_encoder, ax, predictions, target, spikes, title=""):
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
	ax.plot(predictions, label="Prediction")
	ax.plot(target, label="Target")
	print(f"{len(x_scatter_spikes) = }, {np.max(spikes) = }, {np.min(spikes) = }")
	ax.scatter(
		x_scatter_spikes, y=[y_max*1.1] * len(x_scatter_spikes),
		label="Latent space", c='k', marker='|', linewidths=0.5
	)
	# ax.scatter(
	# 	x_scatter_zeros, y=[y_max*1.1] * len(x_scatter_zeros),
	# 	c='k', marker='.', s=0.05
	# )
	ax.set_xlabel("Time [-]")
	ax.set_ylabel("Activity [-]")
	ax.set_title(title)
	ax.legend()


class AutoEncoderTrainingOutput(NamedTuple):
	spikes_auto_encoder: SpikesAutoEncoder
	history: nt.TrainingHistory
	dataset: TimeSeriesAutoEncoderDataset
	

def train_auto_encoder(
		encoder_type: Type[nt.modules.BaseLayer],
		n_units: int,
		n_encoder_steps: int,
		batch_size: int,
		n_iterations: int,
		seed: int = 0
) -> AutoEncoderTrainingOutput:
	params_str = f"{encoder_type.__name__}_{n_units}_{n_encoder_steps}_{batch_size}_{n_iterations}_{seed}"
	checkpoint_folder = f"checkpoints/SpikesAutoEncoder_{params_str}"
	checkpoint_manager = nt.CheckpointManager(checkpoint_folder, metric="train_loss", minimise_metric=True, save_freq=-1)
	spikes_auto_encoder = SpikesAutoEncoder(
		encoder_type, n_units, n_encoder_steps=n_encoder_steps, checkpoint_folder=checkpoint_folder
	).build()
	dataset = TimeSeriesAutoEncoderDataset(sample_size=n_units, seed=seed)
	dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
	trainer = nt.Trainer(
		spikes_auto_encoder,
		# callbacks=[checkpoint_manager]
	)
	history = trainer.train(dataloader, n_iterations=n_iterations, load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR)
	return AutoEncoderTrainingOutput(spikes_auto_encoder=spikes_auto_encoder, history=history, dataset=dataset)


if __name__ == '__main__':
	for s_type in [nt.SpyLIFLayer, nt.LIFLayer, nt.ALIFLayer]:
		for n_u in [
			# 2, 16,
			128,
			# 256, 1024
		]:
			for n_t in [
				# 2, 4,
				# 8,
				# 16,
				32,
				# 64, 128, 256, 512, 1024
			]:
				auto_encoder_training_output = train_auto_encoder(
					encoder_type=s_type,
					n_units=n_u,
					n_encoder_steps=n_t,
					batch_size=256,
					n_iterations=256,
				)
				show_prediction(
					auto_encoder_training_output.dataset.data,
					auto_encoder_training_output.spikes_auto_encoder
				)
				# auto_encoder_training_output.history.plot(show=True)



