import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader

import neurotorch as nt
from dataset import WSDataset
from neurotorch import WilsonCowanLayer
from neurotorch.metrics import RegressionMetrics
from neurotorch.regularization.connectome import DaleLawL2
from neurotorch.visualisation.time_series_visualisation import *


def train_with_params(
		true_time_series: np.ndarray or torch.Tensor,
		learning_rate: float = 1e-3,
		epochs: int = 100,
		forward_weights: Optional[torch.Tensor or np.ndarray] = None,
		std_weights: float = 1,
		dt: float = 1e-3,
		mu: Optional[float or torch.Tensor or np.ndarray] = 0.0,
		mean_mu: Optional[float] = 0.0,
		std_mu: Optional[float] = 1.0,
		r: Optional[float or torch.Tensor or np.ndarray] = 1.0,
		mean_r: Optional[float] = 1.0,
		std_r: Optional[float] = 1.0,
		tau: float = 1.0,
		learn_mu: bool = False,
		learn_r: bool = False,
		learn_tau: bool = False,
		device: torch.device = torch.device("cpu"),
):
	if not torch.is_tensor(true_time_series):
		true_time_series = torch.tensor(true_time_series, dtype=torch.float32, device=device)
	x = true_time_series.T[np.newaxis, :]
	ws_layer = WilsonCowanLayer(
		x.shape[-1], x.shape[-1],
		forward_weights=forward_weights,
		std_weights=std_weights,
		dt=dt,
		r=r,
		mean_r=mean_r,
		std_r=std_r,
		mu=mu,
		mean_mu=mean_mu,
		std_mu=std_mu,
		tau=tau,
		learn_r=learn_r,
		learn_mu=learn_mu,
		learn_tau=learn_tau,
		device=device,
		name="WilsonCowan"
	)
	model = nt.SequentialModel(layers=[ws_layer], device=device, foresight_time_steps=x.shape[1] - 1)
	model.build()
	regularisation = DaleLawL2([ws_layer.forward_weights], alpha=0,
							   reference_weights=nt.init.dale_(torch.zeros(200, 200), inh_ratio=0.5, rho=0.99))
	optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, maximize=True, weight_decay=0.1)
	optimizer_regul = torch.optim.SGD(regularisation.parameters(), lr=5e-4)
	criterion = nn.MSELoss()

	with torch.no_grad():
		W0 = ws_layer.forward_weights.clone()
		plt.imshow(W0.numpy(), cmap="RdBu_r")
		plt.show()
		mu0 = ws_layer.mu.clone()
		r0 = ws_layer.r.clone()
		tau0 = ws_layer.tau.clone()

	dataset = WSDataset(true_time_series.T)
	trainer = nt.trainers.RegressionTrainer(
		model,
		optimizer=optimizer,
		regularization_optimizer=optimizer_regul,
		criterion=lambda pred, y: RegressionMetrics.compute_p_var(y_true=y, y_pred=pred, reduction='mean'),
		regularization=regularisation,
		metrics=[regularisation],
	)
	trainer.train(
		DataLoader(dataset, shuffle=False, num_workers=0),
		n_iterations=epochs,
		exec_metrics_on_train=True,
	)

	x_pred = []
	x_pred.append(torch.unsqueeze(x[:, 0].clone(), dim=1).to(model.device))
	x_pred.append(model.get_prediction_trace(x_pred[0]))
	x_pred = torch.concat(x_pred, dim=1)
	mse_loss = criterion(x_pred, x)
	loss = 1 - mse_loss / torch.var(x)

	out = {}
	out["pVar"] = loss.detach().item()
	out["W"] = ws_layer.forward_weights.detach().numpy()
	out["mu"] = ws_layer.mu.detach().numpy()
	out["r"] = ws_layer.r.detach().numpy()
	out["W0"] = W0.numpy()
	out["mu0"] = mu0.numpy()
	out["r0"] = r0.numpy()
	out["tau0"] = tau0.numpy()
	out["tau"] = ws_layer.tau.detach().numpy()
	out["x_pred"] = torch.squeeze(x_pred).detach().numpy().T

	return out


ts = np.load('timeSeries_2020_12_16_cr3_df.npy')
n_neurons, n_shape = ts.shape
sample_size = 200
sample = np.random.randint(n_neurons, size=sample_size)
data = ts[sample, :]

sigma = 30

for neuron in range(data.shape[0]):
	data[neuron, :] = gaussian_filter1d(data[neuron, :], sigma=sigma)
	data[neuron, :] = data[neuron, :] - np.min(data[neuron, :])
	data[neuron, :] = data[neuron, :] / np.max(data[neuron, :])

forward_weights = nt.init.dale_(torch.zeros(200, 200), inh_ratio=0.5, rho=0.99)

res = train_with_params(
	true_time_series=data,
	learning_rate=1e-2,
	epochs=750,
	forward_weights=forward_weights,
	std_weights=1,
	dt=0.02,
	mu=0.0,
	mean_mu=0,
	std_mu=1,
	r=0.1,
	mean_r=0.2,
	std_r=0,
	tau=0.1,
	learn_mu=True,
	learn_r=True,
	learn_tau=True,
	device=torch.device("cpu")
)

plt.imshow(res["W0"], cmap="RdBu_r")
plt.colorbar()
plt.show()
plt.imshow(res["W"], cmap="RdBu_r")
plt.colorbar()
plt.show()

error = (res["x_pred"] - data) ** 2
plt.plot(error.T)
plt.xlabel("Time [-]")
plt.ylabel("Error L2 [-]")
plt.title(f"pVar: {res['pVar']:.4f}")
plt.show()

VisualiseKMeans(data, nt.Size([nt.Dimension(200, nt.DimensionProperty.NONE, "Neuron [-]"),
							   nt.Dimension(406, nt.DimensionProperty.TIME, "time [s]")])).heatmap(show=True)
VisualiseKMeans(res["x_pred"], nt.Size([nt.Dimension(200, nt.DimensionProperty.NONE, "Neuron [-]"),
										nt.Dimension(406, nt.DimensionProperty.TIME, "time [s]")])).heatmap(show=True)
Visualise(res["x_pred"], nt.Size([nt.Dimension(200, nt.DimensionProperty.NONE, "Neuron [-]"),
								  nt.Dimension(406, nt.DimensionProperty.TIME, "time [s]")])).animate(time_interval=0.1,
																									  forward_weights=
																									  res["W"], dt=0.1,
																									  show=True)

for i in range(sample_size):
	plt.plot(data[i, :], label="True")
	plt.plot(res["x_pred"][i, :], label="Pred")
	plt.ylim([0, 1])
	plt.legend()
	plt.show()
