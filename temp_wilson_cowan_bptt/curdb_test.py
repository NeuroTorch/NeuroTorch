import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from Library_neurotorch import *
import matplotlib.pyplot as plt
from neurotorch.visualisation.time_series_visualisation import *
import torch

ts = np.load('1541_F_dff.npy')
ts = ts[:, 0:1000]
# Smoothing and z-scoring
sigma = 50
(N, T) = ts.shape
filtered_ts = np.zeros((N, T))
for i in range(N):
	new_ts = zscore(ts[i, :])
	# new_ts = new_ts-np.min(new_ts)
	new_ts = new_ts / np.max(np.abs(new_ts))
	new_ts = gaussian_filter1d(new_ts, sigma)
	filtered_ts[i, :] = new_ts

new_filter = np.zeros((filtered_ts.shape[0], filtered_ts.shape[1]))

for n in range(172):
	new_filter[n, :] = filtered_ts[n, :] - np.min(filtered_ts[n, :])
	new_filter[n, :] /= np.max(new_filter[n, :])


res = train_with_params(
	true_time_series=new_filter,
	learning_rate=1e-2,
	epochs=1000,
	forward_weights=None,
	std_weights=1,
	dt=1e-2,
	mu=0.0,
	mean_mu=0,
	std_mu=1,
	r=0,
	mean_r=1,
	std_r=0.1,
	tau=0.8,
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

VisualiseKMeans(new_filter).heatmap(show_axis=False, v=(-0.25, 1.25))
VisualiseKMeans(res["x_pred"]).heatmap(show_axis=False, v=(-0.25, 1.25))
Visualise(res["x_pred"]).animate(time_interval=0.1, forward_weights=res["W"], dt=0.1)

for i in range(172):
	plt.plot(new_filter[i, :], label="True")
	plt.plot(res["x_pred"][i, :], label="Pred")
	plt.ylim([0, 1])
	plt.legend()
	plt.show()