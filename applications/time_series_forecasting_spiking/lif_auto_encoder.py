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
			use_recurrent_connection=False,
			name='encoder'
		).build()
		spikes_decoder = torch.nn.Sequential(
			torchvision.ops.Permute([0, 2, 1]),
			torch.nn.Conv1d(
				n_units, n_units, n_encoder_steps,
				stride=n_encoder_steps,
				bias=False
			),
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
	

def show_prediction(time_series, auto_encoder: SpikesAutoEncoder):
	target = nt.to_tensor(time_series, dtype=torch.float32)
	
	out = auto_encoder(torch.unsqueeze(target, dim=1))[0]
	if isinstance(out, dict):
		out = out[list(out.keys())[0]]
	
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
	
	axes[1].plot(predictions[indices[0]], label="Prediction")
	axes[1].plot(target[indices[0]], label="Target")
	axes[1].set_xlabel("Time [-]")
	axes[1].set_ylabel("Activity [-]")
	axes[1].set_title("Best Prediction vs Target")
	axes[1].legend()
	
	axes[2].plot(predictions[indices[-1]], label="Prediction")
	axes[2].plot(target[indices[-1]], label="Target")
	axes[2].set_xlabel("Time [-]")
	axes[2].set_ylabel("Activity [-]")
	axes[2].set_title("Worst Prediction vs Target")
	axes[2].legend()
	
	fig.set_tight_layout(True)
	plt.show()
	

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
	trainer = nt.Trainer(spikes_auto_encoder, callbacks=[checkpoint_manager])
	history = trainer.train(dataloader, n_iterations=n_iterations, load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR)
	return AutoEncoderTrainingOutput(spikes_auto_encoder=spikes_auto_encoder, history=history, dataset=dataset)


if __name__ == '__main__':
	for s_type in [nt.SpyLIFLayer, nt.LILayer, nt.ALIFLayer]:
		for n_u in [2, 16, 128, 256, 1024]:
			for n_t in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
				auto_encoder_training_output = train_auto_encoder(
					encoder_type=s_type,
					n_units=n_u,
					n_encoder_steps=n_t,
					batch_size=256,
					n_iterations=256,
				)
				# show_prediction(
				# 	auto_encoder_training_output.dataset.data,
				# 	auto_encoder_training_output.spikes_auto_encoder
				# )
				# auto_encoder_training_output.history.plot(show=True)



