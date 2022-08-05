import torch
import tqdm
from matplotlib import pyplot as plt

import neurotorch as nt
from applications.time_series_forecasting_spiking.dataset import TimeSeriesDataset
from applications.time_series_forecasting_spiking.lif_auto_encoder import train_auto_encoder, show_prediction
from neurotorch.transforms import ConstantValuesTransform

n_units = 2
n_encoder_steps = 32
n_iterations = 300

ws_ts = TimeSeriesDataset(
	input_transform=ConstantValuesTransform(
		n_steps=n_encoder_steps,
	),
	# target_transform=ConstantValuesTransform(
	# 	n_steps=n_encoder_steps,
	# ),
	sample_size=n_units,
)

auto_encoder_training_output = train_auto_encoder(
	n_units=n_units, n_encoder_steps=n_encoder_steps, batch_size=256, n_iterations=256
)
show_prediction(auto_encoder_training_output.dataset.data, auto_encoder_training_output.spikes_auto_encoder)
# auto_encoder_training_output.history.plot(show=True)


spikes_encoder = auto_encoder_training_output.spikes_auto_encoder.spikes_encoder
spikes_encoder.learning_type = nt.LearningType.NONE
spikes_encoder.requires_grad_(False)
spikes_decoder = auto_encoder_training_output.spikes_auto_encoder.spikes_decoder
spikes_decoder.requires_grad_(False)

lif_layer = nt.SpyLIFLayer(
	input_size=nt.Size(
		[
			nt.Dimension(None, nt.DimensionProperty.TIME),
			nt.Dimension(n_units, nt.DimensionProperty.NONE)
		]
	),
	output_size=n_units,
	use_recurrent_connection=False,
).build()
all_parameters = [
	*list(lif_layer.parameters()),
	# *list(spikes_encoder.parameters()),
	*list(spikes_decoder.parameters()),
]
optimizer = torch.optim.Adam(all_parameters, lr=1e-3, maximize=True)
criterion = nt.losses.PVarianceLoss()

t0, target = ws_ts[0]
t0 = torch.unsqueeze(t0, 0)
target = torch.unsqueeze(target, 0)


def predict():
	t0_spikes = []
	hh_encoder = None
	for t in range(n_encoder_steps):
		x = t0[:, t]
		x, hh_encoder = spikes_encoder(x, hh_encoder)
		t0_spikes.append(x)
	t0_spikes = torch.stack(t0_spikes, dim=1)
	t0_decode = spikes_decoder(t0_spikes)
	
	spikes_preds = []
	hh = None
	for t in range(n_encoder_steps):
		x = t0_spikes[:, t]
		x, hh = lif_layer(x, hh)
		spikes_preds.append(x)
	
	for t in range((ws_ts.n_time_steps - 1) * n_encoder_steps):
		x = spikes_preds[-1]
		x, hh = lif_layer(x, hh)
		spikes_preds.append(x)
	
	ts_decode = spikes_decoder(torch.stack(spikes_preds, dim=1)[:, n_encoder_steps:])
	preds = torch.concat([t0_decode, ts_decode], dim=1)
	# preds = spikes_decoder(torch.stack(spikes_preds, dim=1))
	return preds


def predict_spikes():
	t0_spikes = []
	hh_encoder = None
	for t in range(n_encoder_steps):
		x = t0[:, t]
		x, hh_encoder = spikes_encoder(x, hh_encoder)
		t0_spikes.append(x)
	t0_spikes = torch.stack(t0_spikes, dim=1)
	t0_decode = spikes_decoder(t0_spikes)
	
	spikes_preds = []
	hh = None
	for t in range(n_encoder_steps):
		x = t0_spikes[:, t]
		x, hh = lif_layer(x, hh)
		spikes_preds.append(x)
	
	for t in range((ws_ts.n_time_steps - 1) * n_encoder_steps):
		x = spikes_preds[-1]
		x, hh = lif_layer(x, hh)
		spikes_preds.append(x)
	
	ts_decode = spikes_decoder(torch.stack(spikes_preds, dim=1)[:, n_encoder_steps:])
	preds = torch.concat([t0_decode, ts_decode], dim=1)
	# preds = spikes_decoder(torch.stack(spikes_preds, dim=1))
	return preds


def train():
	p_bar = tqdm.tqdm(range(n_iterations))
	for i in p_bar:
		preds = predict()
		loss = criterion(preds, target.to(preds.device))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		p_bar.set_description(f"pVar: {loss.detach().cpu().item():.4f}")
		if loss.detach().cpu().item() > 0.98:
			p_bar.close()
			break

# predictions = predict().detach().cpu()
# errors = torch.squeeze(predictions - target.to(predictions.device))**2
# mse_loss = criterion(predictions, target.to(predictions.device))
# pVar = 1 - mse_loss / torch.var(target.to(mse_loss.device))
# fig, ax = plt.subplots(1, 1, figsize=(15, 8))
# ax.plot(errors.detach().cpu().numpy())
# ax.set_xlabel("Time [-]")
# ax.set_ylabel("Error L2 [-]")
# ax.set_title(f"pVar: {pVar.detach().cpu().item():.4f}")
# # ax.legend()
# plt.show()
#
#
# mean_errors = torch.mean(errors, dim=0)
# mean_error_sort, indices = torch.sort(mean_errors)
# predictions = torch.squeeze(predictions).numpy().T
# target = torch.squeeze(target).numpy().T
# # for rank in [0, 1, -2, -1]:
# for rank in range(n_units):
# 	unit_idx = indices[rank]
# 	fig, ax = plt.subplots(1, 1, figsize=(15, 8))
# 	ax.plot(predictions[unit_idx], label=f"Prediction")
# 	ax.plot(target[unit_idx], label=f"Target")
# 	ax.set_title(f"Unit {unit_idx}, rank: {rank}")
# 	ax.set_xlabel("Time [-]")
# 	ax.set_ylabel("Activity [-]")
# 	ax.legend()
# 	plt.show()
# 	plt.close(fig)


if __name__ == '__main__':
	train()
	
	predictions = torch.squeeze(predict().detach().cpu())
	target = torch.squeeze(target.detach().cpu())
	errors = torch.squeeze(predictions - target.to(predictions.device))**2
	mse_loss = torch.nn.MSELoss()(predictions, target.to(predictions.device))
	pVar = 1 - mse_loss / torch.var(target.to(mse_loss.device))
	fig, axes = plt.subplots(3, 1, figsize=(15, 8))
	axes[0].plot(errors.detach().cpu().numpy())
	axes[0].set_xlabel("Time [-]")
	axes[0].set_ylabel("Squared Error [-]")
	axes[0].set_title(f"pVar: {pVar.detach().cpu().item():.4f}")
	
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
