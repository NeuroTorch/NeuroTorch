from Library import *
from scipy.ndimage import gaussian_filter1d
import numpy as np
import torch
import matplotlib.pyplot as plt
from neurotorch.visualisation.time_series_visualisation import *
import scipy.signal

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


forward_weights = random_matrix(200, 0.2)


res = train_with_params(
	true_time_series=data,
	learning_rate=1e-2,
	epochs=1000,
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

plt.imshow(res["W0"], cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar()
plt.show()
plt.imshow(res["W"], cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar()
plt.show()

print(res["r"])

print(res["tau"])

error = (res["x_pred"] - data) ** 2
plt.plot(error.T)
plt.xlabel("Time [-]")
plt.ylabel("Error L2 [-]")
plt.title(f"pVar: {res['pVar']:.4f}")
plt.show()

VisualiseKMeans(data).heatmap(show_axis=False)
VisualiseKMeans(res["x_pred"]).heatmap(show_axis=False)
Visualise(res["x_pred"]).animate(time_interval=0.1, forward_weights=res["W"], dt=0.1)


train_dynamic = WilsonCowanTimeSeries(
	forward_weights=torch.tensor(res["W"]),
	time_step=res["x_pred"].shape[1],
	t_0=torch.tensor(data[:, 0]),
	dt=0.1,
	r=torch.tensor(res["r"]),
	mu=torch.tensor(res["mu"]),
	tau=torch.tensor(res["tau"])
)

# train_time_series = train_dynamic.compute()



for i in range(sample_size):
	plt.plot(data[i, :], label="True")
	plt.plot(res["x_pred"][i, :], label="Pred")
	plt.ylim([0, 1])
	plt.legend()
	plt.show()