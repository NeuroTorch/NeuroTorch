import time

import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt

import neurotorch as nt
from applications.time_series_forecasting_spiking.dataset import TimeSeriesDataset
from applications.time_series_forecasting_spiking.lif_auto_encoder import train_auto_encoder, show_prediction, \
	show_single_preds
from neurotorch.transforms import ConstantValuesTransform


def create_dataset(auto_encoder_training_output, n_time_steps=None):
	ws_ts = TimeSeriesDataset(
		input_transform=ConstantValuesTransform(
			n_steps=auto_encoder_training_output.spikes_auto_encoder.n_encoder_steps,
		),
		units=auto_encoder_training_output.dataset.units_indexes,
		n_time_steps=n_time_steps,
	)
	return ws_ts


def create_network(n_units, n_encoder_steps, encoder_type, n_aux_units=0):
	auto_encoder_training_output = train_auto_encoder(
		encoder_type=encoder_type, n_units=n_units+n_aux_units, n_encoder_steps=n_encoder_steps
	)
	# show_prediction(auto_encoder_training_output.dataset.data, auto_encoder_training_output.spikes_auto_encoder)
	# auto_encoder_training_output.history.plot(show=True)
	
	spikes_auto_encoder = auto_encoder_training_output.spikes_auto_encoder
	spikes_encoder = auto_encoder_training_output.spikes_auto_encoder.spikes_encoder
	spikes_encoder.learning_type = nt.LearningType.NONE
	spikes_encoder.requires_grad_(False)
	spikes_decoder = auto_encoder_training_output.spikes_auto_encoder.spikes_decoder
	spikes_decoder.requires_grad_(False)
	
	lif_layer = encoder_type(
		input_size=nt.Size(
			[
				nt.Dimension(None, nt.DimensionProperty.TIME),
				nt.Dimension(n_units+n_aux_units, nt.DimensionProperty.NONE)
			]
		),
		output_size=n_units+n_aux_units,
		use_recurrent_connection=True,
		learning_type=nt.LearningType.BPTT,
		name="predictor",
	).build()
	
	return lif_layer, auto_encoder_training_output


def create_sequential_network(n_units, n_encoder_steps, encoder_type, n_aux_units=0):
	lif_layer, auto_encoder_training_output = create_network(n_units, n_encoder_steps, encoder_type, n_aux_units)
	seq = nt.SequentialModel(
		input_transform=[auto_encoder_training_output.spikes_auto_encoder.spikes_encoder],
		layers=[lif_layer],
		output_transform=[auto_encoder_training_output.spikes_auto_encoder.spikes_decoder],
	)


def predict(network, spikes_auto_encoder, dataset=None, n_tbptt_steps=-1, criterion=None):
	t0, target = dataset[0]
	t0 = torch.unsqueeze(t0, 0)
	target = torch.unsqueeze(target, 0)
	
	n_encoder_steps = spikes_auto_encoder.n_encoder_steps
	n_units = dataset.n_units
	n_aux_units = int(network.input_size) - n_units
	
	if n_aux_units > 0:
		t0 = torch.cat([t0, torch.randn((t0.shape[0], n_aux_units)).repeat(1, t0.shape[1], 1).to(t0.device)], dim=-1)
	t0_spikes = spikes_auto_encoder.encode(t0)
	# if n_aux_units > 0:
	# 	t0_spikes = torch.cat([
	# 		t0_spikes,
	# 		torch.zeros((t0_spikes.shape[0], t0_spikes.shape[1], n_aux_units)).to(t0_spikes.device)
	# 	], dim=-1)
	
	spikes_preds = []
	x, hh = None, None
	for t in range(n_encoder_steps):
		x = t0_spikes[:, t]
		x, hh = network(x, hh)
	
	spikes_preds = torch.unsqueeze(x, 1)
	T = (dataset.n_time_steps - 1) * n_encoder_steps - 1
	for t in range(T):
		x, hh = network(x, hh)
		# spikes_preds.append(x)
		spikes_preds = torch.concat([spikes_preds, torch.unsqueeze(x, 1)], dim=1)
		
		do_tbptt = n_tbptt_steps > 0
		in_range_of_tbptt = 0 < t < T - n_encoder_steps + 1
		if do_tbptt and in_range_of_tbptt and len(spikes_preds) % (n_tbptt_steps * n_encoder_steps) == 0:
			local_preds = spikes_auto_encoder.decode(spikes_preds[:, -n_tbptt_steps * n_encoder_steps:, :n_units])
			t_start, t_stop = t // (n_tbptt_steps * n_encoder_steps), t // ((n_tbptt_steps + 1) * n_encoder_steps)
			local_targets = target[:, t_start:t_stop]
			loss = criterion(local_preds, local_targets.to(local_preds.device))
			loss.backward()
			loss.detach_()
	
	# spikes_preds = torch.stack(spikes_preds, dim=1)
	spikes_preds = torch.concat([t0_spikes, spikes_preds], dim=1)
	preds = spikes_auto_encoder.decode(spikes_preds)[:, :, :n_units]
	return preds, spikes_preds


def predict_from_sequential(network, dataset, *args):
	pass


def train(network, spikes_auto_encoder, dataset, n_iterations=256, desc=""):
	loss_schedule = np.linspace(0.0, 0.98, num=5)
	curr_stage = 0
	curr_lr = 5e-5
	lr_gamma = 0.5
	pVar_coeff = 1.0
	history = nt.callbacks.TrainingHistory()
	
	all_parameters = [
		*list(network.parameters()),
		*list(spikes_auto_encoder.spikes_encoder.parameters()),
		*list(spikes_auto_encoder.spikes_decoder.parameters()),
	]
	criterion = nt.losses.PVarianceLoss()
	optimizer = torch.optim.AdamW(all_parameters, lr=curr_lr, maximize=True, weight_decay=0.0)
	
	t0, target = dataset[0]
	target = torch.unsqueeze(target, 0)
	
	p_bar = tqdm.tqdm(range(n_iterations), desc=desc)
	for i in p_bar:
		optimizer.zero_grad()
		preds, spikes_preds = predict(network, spikes_auto_encoder, dataset=dataset, n_tbptt_steps=-1, criterion=criterion)
		target_spikes = spikes_auto_encoder.encode(torch.permute(target, (1, 0, 2))).view(1, -1, target.shape[-1])
		pVar = criterion(preds, target.to(preds.device))
		latent_pVar = criterion(spikes_preds, target_spikes.to(spikes_preds.device))
		loss = pVar_coeff * pVar + (1 - pVar_coeff) * latent_pVar
		loss.backward()
		optimizer.step()
		metrics = {
			"pVar": f"{pVar.detach().cpu().item():.4f}",
			"latent_pVar": f"{latent_pVar.detach().cpu().item():.4f}",
			"loss": f"{loss.detach().cpu().item():.4f}",
			"lr": f"{curr_lr:.4e}"
		}
		p_bar.set_postfix(metrics)
		history.insert(i, {k: float(v) for k, v in metrics.items()})
		if loss.detach().cpu().item() > 0.98:
			p_bar.close()
			break
		if curr_stage < len(loss_schedule) and loss.detach().cpu().item() > loss_schedule[curr_stage]:
			curr_stage += 1
			curr_lr *= lr_gamma
			for g in optimizer.param_groups:
				g['lr'] = curr_lr
	return history


def visualize_single_training(history: nt.TrainingHistory, network, spikes_auto_encoder, dataset):
	t0, target = dataset[0]
	target = torch.unsqueeze(target, 0)
	history.plot(show=True)
	
	preds, spikes_preds = predict(network, spikes_auto_encoder, dataset)
	spikes = torch.squeeze(spikes_preds).detach().cpu().numpy()
	predictions = torch.squeeze(preds.detach().cpu())
	target = torch.squeeze(target.detach().cpu())
	errors = torch.squeeze(predictions - target.to(predictions.device))**2
	mse_loss = torch.nn.MSELoss()(predictions, target.to(predictions.device))
	pVar = 1 - mse_loss / torch.var(target.to(mse_loss.device))
	fig, axes = plt.subplots(3, 1, figsize=(12, 8))
	axes[0].plot(errors.detach().cpu().numpy())
	axes[0].set_xlabel("Time [-]")
	axes[0].set_ylabel("Squared Error [-]")
	axes[0].set_title(
		f"Predictor: {spikes_auto_encoder.encoder_type.__name__}"
		f"<{spikes_auto_encoder.n_units}u, {spikes_auto_encoder.n_encoder_steps}t>, "
		f"pVar: {pVar.detach().cpu().item():.3f}, "
	)
	
	mean_errors = torch.mean(errors, dim=0)
	mean_error_sort, indices = torch.sort(mean_errors)
	predictions = torch.squeeze(predictions).numpy().T
	target = torch.squeeze(target).numpy().T
	
	best_idx, worst_idx = indices[0], indices[-1]
	spikes = spikes.reshape(-1, spikes_auto_encoder.n_encoder_steps, spikes_auto_encoder.n_units)
	show_single_preds(
		spikes_auto_encoder, axes[1], predictions[best_idx], target[best_idx], spikes[:, :, best_idx],
		title="Best Prediction"
	)
	show_single_preds(
		spikes_auto_encoder, axes[2], predictions[worst_idx], target[worst_idx], spikes[:, :, worst_idx],
		title="Worst Prediction"
	)
	
	fig.set_tight_layout(True)
	plt.show()
	plt.close(fig)


def run_p_var_vs_n_encoder_steps(n_encoder_steps_space=None):
	if n_encoder_steps_space is None:
		last_power = 8
		n_encoder_steps_space = 2**np.arange(1, last_power + 1)
	
	p_var_history = []
	n_encoder_steps_history = []
	time_history = []
	
	for i, n_encoder_steps in enumerate(n_encoder_steps_space):
		n_units = 128
		n_iterations = 256
		encoder_type = nt.SpyLIFLayer
		
		lif_layer, auto_encoder_training_output = create_network(n_units, n_encoder_steps, encoder_type)
		dataset = create_dataset(auto_encoder_training_output, n_time_steps=4)
		start_time = time.time()
		hist, lr_hist = train(
			lif_layer, auto_encoder_training_output.spikes_auto_encoder, dataset,
			n_iterations=n_iterations,
			desc=f"n_encoder_steps: {n_encoder_steps}"
		)
		time_history.append(time.time() - start_time)
		p_var_history.append(hist[-1])
		n_encoder_steps_history.append(n_encoder_steps)
		
		fig, axes = plt.subplots(2, 1, figsize=(12, 8))
		if not isinstance(axes, np.ndarray):
			axes = np.asarray([axes])
		axes[0].plot(n_encoder_steps_history, p_var_history, label="Loss")
		axes[0].set_xlabel("Encoder Steps [-]")
		axes[0].set_ylabel("Loss [-]")
		axes[0].legend()
		
		axes[1].plot(n_encoder_steps_history, time_history, label="Time")
		axes[1].set_xlabel("Encoder Steps [-]")
		axes[1].set_ylabel("Time [s]")
		axes[1].legend()
		
		plt.savefig("p_var_vs_n_encoder_steps.png", dpi=300)
		if i != len(n_encoder_steps_space) - 1:
			plt.close(fig)
	plt.show()


def main():
	n_units = 128
	n_aux_units = 0
	n_encoder_steps = 32
	n_iterations = 4096
	n_time_steps = 16
	encoder_type = nt.SpyLIFLayer
	
	lif_layer, auto_encoder_training_output = create_network(n_units, n_encoder_steps, encoder_type, n_aux_units)
	# show_prediction(auto_encoder_training_output.dataset.data, auto_encoder_training_output.spikes_auto_encoder)
	dataset = create_dataset(auto_encoder_training_output, n_time_steps=n_time_steps)
	start_time = time.time()
	history = train(
		lif_layer, auto_encoder_training_output.spikes_auto_encoder, dataset,
		n_iterations=n_iterations,
		desc=f"Training [encoder: {encoder_type.__name__}, encoder_steps: {n_encoder_steps}, time_steps: {n_time_steps},"
		f" units: {n_units}, aux_units: {n_aux_units}]"
	)
	print(f"Elapsed time: {time.time() - start_time :.2f} seconds")
	visualize_single_training(history, lif_layer, auto_encoder_training_output.spikes_auto_encoder, dataset)


if __name__ == '__main__':
	main()
