import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import neurotorch as nt
from neurotorch.metrics import RegressionMetrics
from neurotorch.visualisation.time_series_visualisation import Visualise, VisualiseKMeans
from neurotorch.regularization.connectome import DaleLawL2
from neurotorch import WilsonCowanLayer
from src.neurotorch.transforms.base import to_tensor
from typing import *

class WCDataset(Dataset):
	def __init__(self, x):
		self.x = x

	def __len__(self):
		return 1

	def __getitem__(self, item):
		return torch.unsqueeze(self.x[0], dim=0), self.x[1:]


import numpy as np

time_series = np.load('timeSeries_2020_12_16_cr3_df.npy')
n_neurons, n_shape = time_series.shape
sample_size = 200
sample = np.random.randint(n_neurons, size=sample_size)
data = time_series[sample, :]

from scipy.ndimage import gaussian_filter1d

sigma = 20

for neuron in range(data.shape[0]):
	data[neuron, :] = gaussian_filter1d(data[neuron, :], sigma=sigma)
	data[neuron, :] = data[neuron, :] - np.min(data[neuron, :])
	data[neuron, :] = data[neuron, :] / np.max(data[neuron, :])


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
    return torch.tensor(W, dtype=torch.float32, device=torch.device("cpu")).T  # Transpose so it follow our convention (neuron i -> neuron j)


def train_with_params(
		true_time_series: np.ndarray or torch.Tensor,
		learning_rate: float = 1e-2,
		epochs: int = 100,
		forward_weights: Optional[torch.tensor or np.ndarray] = None,
		std_weights: float = 1.0,
		dt: float = 0.02,
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
		device: torch.device = torch.device("cpu")
):
	if not torch.is_tensor(true_time_series):  # convert time series to torch tensor
		true_time_series = torch.tensor(true_time_series, dtype=torch.float32, device=device)
	ts = true_time_series.T  # ensure that time series has format (1 x T x N)
	wc_layer = WilsonCowanLayer(
		ts.shape[-1], ts.shape[-1],
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

	model = nt.SequentialModel(layers=[wc_layer], device=device, foresight_time_steps=ts.shape[0] - 1)
	model.build()

	optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, maximize=True, weight_decay=0.01)  # Weight decay can be applied to reduce the energy if connectome
	regularisation = DaleLawL2([wc_layer.forward_weights], alpha=0.8, reference_weights=random_matrix(ts.shape[-1], 0.99))
	optimizer_regul = torch.optim.SGD(regularisation.parameters(), lr=5e-4)
	dataset = WCDataset(ts)
	trainer = nt.trainers.RegressionTrainer(
		model,
		optimizer=optimizer,
		regularization_optimizer=optimizer_regul,
		criterion=lambda pred, true: RegressionMetrics.compute_p_var(y_true=true, y_pred=pred),
		regularisation=regularisation,
		metrics=[regularisation]
	)
	trainer.train(
		DataLoader(dataset, shuffle=False, num_workers=0),
		n_iterations=epochs,
		exec_metrics_on_train=True
	)
	return "done"


initial_forward_weight = random_matrix(200, 0.99)

result = train_with_params(
	true_time_series=data,
	learning_rate=1e-2,
	epochs=500,
	forward_weights=initial_forward_weight,
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