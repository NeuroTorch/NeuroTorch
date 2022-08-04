from typing import *

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from neurotorch.transforms import to_tensor
from neurotorch.visualisation.time_series_visualisation import *
from neurotorch.regularization.connectome import DaleLaw

def random_matrix(N, rho):
    """Half excitatory, half inhibitory."""
    W = np.zeros((N, N))
    i, j = np.triu_indices(N, 1)
    N_0 = int((1 - rho) * len(i)) # Number of zero values
    valuesUpper = np.append(np.array([0] * N_0),
    np.random.normal(0, (1 / np.sqrt(N * rho * (1 - rho))), (len(i) - N_0, )))
    valuesLower = np.append(np.array([0] * N_0),
    np.random.normal(0, (1 / np.sqrt(N * rho * (1 - rho))), (len(i) - N_0, )))
    np.random.shuffle(valuesUpper)
    np.random.shuffle(valuesLower)
    W[i, j] = valuesUpper
    W[j, i] = valuesLower
    W = np.abs(W)
    W[:, int(N / 2):] *= -1
    return torch.tensor(W, dtype=torch.float32, device=torch.device("cpu"))


class WilsonCowanDynamic(nn.Module):

	def __init__(
			self,
			connectome_size: Optional[int] = None,
			forward_weights: Optional[torch.Tensor or np.ndarray] = None,
			std_weights: Optional[float] = 1,
			dt: float = 1e-3,
			r: float or torch.Tensor or np.ndarray = 1.0,
			mean_r: float = 0.0,
			std_r: float = 0.0,
			mu: float or torch.Tensor or np.ndarray = 0.0,
			mean_mu: float = 0.0,
			std_mu: float = 0.0,
			learn_r: bool = False,
			learn_mu: bool = False,
			tau: float = 1.0,
			learn_tau=False,
			device: torch.device = torch.device('cpu')
	):
		super(WilsonCowanDynamic, self).__init__()
		self.connectome_size = connectome_size
		self.forward_weights = forward_weights
		self.std_weights = std_weights
		self.device = device
		self.dt = dt
		self.r_sqrt = torch.sqrt(to_tensor(r, dtype=torch.float32)).to(self.device)
		self.mean_r = mean_r
		self.std_r = std_r
		self.mu = mu
		if not torch.is_tensor(self.mu):
			self.mu = torch.tensor(self.mu, dtype=torch.float32, device=self.device)
		self.mean_mu = mean_mu
		self.std_mu = std_mu
		self.learn_r = learn_r
		self.learn_mu = learn_mu
		self.tau = torch.tensor(tau, dtype=torch.float32, device=self.device)
		self.learn_tau = learn_tau

	@property
	def r(self):
		return self.r_sqrt**2

	def build(self):
		if self.forward_weights is not None:
			self.forward_weights = torch.tensor(self.forward_weights, dtype=torch.float32, device=self.device)
		else:
			self.forward_weights = torch.empty(self.connectome_size, self.connectome_size, dtype=torch.float32,
											   device=self.device)
			torch.nn.init.normal_(self.forward_weights, mean=0.0, std=self.std_weights)
		self.forward_weights = torch.nn.Parameter(self.forward_weights, requires_grad=True)
		if self.learn_mu:
			self.mu = torch.empty(self.connectome_size, dtype=torch.float32, device=self.device)
			torch.nn.init.normal_(self.mu, mean=self.mean_mu, std=self.std_mu)
			self.mu = torch.nn.Parameter(self.mu, requires_grad=True)
		if self.learn_r:
			_r = torch.empty(self.connectome_size, dtype=torch.float32, device=self.device)
			torch.nn.init.normal_(_r, mean=self.mean_r, std=self.std_r)
			self.r_sqrt = torch.nn.Parameter(torch.sqrt(torch.abs(_r)), requires_grad=True)
		if self.learn_tau:
			self.tau = torch.nn.Parameter(self.tau, requires_grad=True)

	def forward(self, inputs: torch.Tensor):
		ratio_dt_tau = self.dt / self.tau
		transition_rate = 1 - self.r * inputs
		# sigmoid = torch.sigmoid(self.forward_weights @ inputs - self.mu)
		sigmoid = torch.sigmoid((inputs.T @ self.forward_weights).T - self.mu)
		output = inputs * (1 - ratio_dt_tau) + ratio_dt_tau * transition_rate * sigmoid
		return output

class WilsonCowanTimeSeries:

	def __init__(
			self,
			forward_weights: np.ndarray or torch.Tensor,
			time_step: int,
			t_0: np.ndarray or torch.Tensor,
			dt: float = 1e-3,
			r: float or torch.Tensor or np.ndarray = 1.0,
			mu : float or torch.Tensor or np.ndarray = 0.0,
			tau: float = 1.0,
			device: torch.device = torch.device("cpu")
	):
		self.forward_weights = forward_weights
		self.time_step = time_step
		self.t_0 = t_0
		self.dt = dt
		self.r = r
		self.mu = mu
		self.tau = tau
		self.device = device
		self.dynamic = WilsonCowanDynamic(
			forward_weights=self.forward_weights,
			dt=self.dt,
			r=self.r,
			mu=self.mu,
			tau=self.tau,
			device=self.device
		)

	def compute(self):
		time_series = torch.zeros(self.forward_weights.shape[0], self.time_step, dtype=torch.float32, device=self.device)
		time_series[:, 0] = self.t_0
		for i in range(1, self.time_step):
			time_series[:, i] = self.dynamic(time_series[:, i - 1])
		return time_series


def train_with_params(
		true_time_series: np.ndarray or torch.Tensor,
		learning_rate: float = 1e-3,
		epochs: int = 100,
		forward_weights: Optional[torch.Tensor or np.ndarray] = None,
		std_weights: float = 1,
		dt: float = 1e-3,
		mu: Optional[float or torch.Tensor or np.ndarray] = 0.0,
		mean_mu : Optional[float] = 0.0,
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
	x = true_time_series
	model = WilsonCowanDynamic(
		connectome_size=x.shape[0],
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
	)
	model.build()
	regularisation = DaleLaw(t=0, reference_weights=random_matrix(x.shape[0], 0.99))
	optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, maximize=True, weight_decay=0.001)
	optimizer_regul = torch.optim.SGD([model.forward_weights], lr=5e-4)
	#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, maximize=True)
	criterion = nn.MSELoss()

	with torch.no_grad():
		W0 = model.forward_weights.clone()
		mu0 = model.mu.clone()
		r0 = model.r.clone()
		tau0 = model.tau.clone()

	progress_bar = tqdm(
		range(epochs),
		total=epochs,
		desc="Training",
		unit="epoch",
	)

	for epoch in progress_bar:
		# Forward pass

		x_pred = []
		x_pred.append(x[:, 0].clone())
		forward_tensor = x[:, 0].clone()
		x_pred.append(model(forward_tensor))

		for i in range(1, x.shape[1] - 1):
			forward_tensor = model(forward_tensor)
			x_pred.append(forward_tensor)

			# if truncated is not None:
			# 	if i % truncated == 0:
			# 		mse_loss = criterion(x, x_pred)
			# 		loss = 1 - mse_loss/torch.std(x)

		# Loss
		x_pred = torch.stack(x_pred, dim=1)
		mse_loss = criterion(x_pred, x)
		loss = 1 - mse_loss/torch.var(x)


		# Gradient
		optimizer.zero_grad()
		loss.backward()
		# update
		optimizer.step()

		loss_regul = regularisation(model.forward_weights)
		optimizer_regul.zero_grad()
		loss_regul.backward()
		optimizer_regul.step()

		postfix = dict(
			pVar=f"{loss.detach().item():.5f}",
			MSE=f"{mse_loss.detach().item():.5f}",
			loss_regul=f"{loss_regul.detach().item():.5f}",
		)
		progress_bar.set_postfix(postfix)

		if loss > 0.99:
			break

	out = {}
	out["pVar"] = loss.detach().item()
	out["W"] = model.forward_weights.detach().numpy()
	out["mu"] = model.mu.detach().numpy()
	out["r"] = model.r.detach().numpy()
	out["W0"] = W0.numpy()
	out["mu0"] = mu0.numpy()
	out["r0"] = r0.numpy()
	out["tau0"] = tau0.numpy()
	out["tau"] = model.tau.detach().numpy()
	out["x_pred"] = x_pred.detach().numpy()

	return out


if __name__ == '__main__':
	n_neuron = 172
	W = 1 * torch.randn(n_neuron, n_neuron)
	t_0 = torch.rand(n_neuron,)
	mu = 0
	tau = 1
	r = 0
	dt = 0.1
	dynamic = WilsonCowanTimeSeries(forward_weights=W, t_0=t_0, mu=mu, tau=tau, r=r, time_step=1500, dt=dt)
	time_series = dynamic.compute()

	Visualise(time_series.detach().numpy()).plot_timeseries()



	res = train_with_params(
		true_time_series=time_series,
		learning_rate=1e-2,
		epochs=150,
		forward_weights=None,
		std_weights=1,
		dt=dt,
		mu=mu,
		mean_mu=0,
		std_mu=1,
		r=1,
		mean_r=1,
		std_r=0.1,
		tau=1,
		learn_mu=True,
		learn_r=True,
		learn_tau=True,
		device=torch.device("cpu")
	)

	for i in range(n_neuron):
		plt.plot(time_series[i, :], label="True")
		plt.plot(res["x_pred"][i, :], label="Pred")
		plt.ylim([0, 1])
		plt.legend()
		plt.show()



