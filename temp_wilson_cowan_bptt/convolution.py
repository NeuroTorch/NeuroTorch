from Library_neurotorch import *
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

sigma = 2

radius = int(4 * sigma + 0.5)
gauss = np.exp(-0.5*(np.linspace(-radius, radius + 1, 406)/float(sigma))**2) / (sigma * np.sqrt(2 * np.pi))
#gauss = np.exp(-0.5*(np.linspace(-radius, radius + 1, 406)/float(sigma))**2)
#gauss = gauss / np.sum(gauss)

# CORRELATION
# f_time_g = np.fft.fft(data, axis=1).conj() * np.fft.fft(gauss)
#
# h = np.fft.fft(f_time_g, axis=1)
#
# # CONVOLUTION

f_time_g = np.fft.fft(data, axis=1) * np.fft.fft(gauss)

h = np.fft.ifft(f_time_g, axis=1).real

normalize = []

h_not_normlized = h.copy()

for neuron in range(data.shape[0]):
	minimum = np.min(h[neuron, :])
	h[neuron, :] = h[neuron, :] - minimum
	maximum = np.max(h[neuron, :])
	h[neuron, :] = h[neuron, :] / maximum
	normalize.append((maximum, minimum))

# for neuron in range(data.shape[0]):
# 	data[neuron, :] = gaussian_filter1d(data[neuron, :], sigma=sigma)
# 	data[neuron, :] = data[neuron, :] - np.min(data[neuron, :])
# 	data[neuron, :] = data[neuron, :] / np.max(data[neuron, :])


# for neuron in range(data.shape[0]):
# 	plt.plot(data[neuron, :], label="data")
# 	plt.plot(h[neuron, :], label="h")
# 	plt.legend(shadow=True, fancybox=True)
# 	plt.show()

# TRAIN HERE

forward_weights = random_matrix(200, 0.2)

res = train_with_params(
	true_time_series=h.real,
	learning_rate=1e-2,
	epochs=200,
	forward_weights=forward_weights,
	std_weights=1,
	dt=1e-3,
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
h_pred = res["x_pred"]

h_pred = h

h_pred[3, :] = res["x_pred"][3, :]


# Denormalize:


for neuron in range(data.shape[0]):
	maximum, minimum = normalize[neuron]
	h_pred[neuron, :] = (h_pred[neuron, :] * maximum) + minimum

# for neuron in range(5):
# 	final = np.fft.ifft(np.fft.fft(h_pred[neuron, :]) / np.fft.fft(gauss))
# 	plt.plot(final, label="pred")
# 	plt.plot(data[neuron, :], label="data")
# 	#plt.plot(h_not_normlized[neuron, :], label="h_not_normlized")
# 	#plt.plot(h_pred[neuron, :], label="h_pred")
# 	plt.show()


# DECORRELATION

# final_temp = (np.fft.ifft(h_pred, axis=1) / np.fft.fft(gauss)).conj()
#
# final = np.fft.ifft(final_temp, axis=1).real

# DECONVOLVE

final = np.fft.ifft((np.fft.fft(h_pred + 10, axis=1) / np.fft.fft(gauss)), axis=1).real



for neuron in range(data.shape[0]):
	plt.plot(data[neuron, :], label="True")
	plt.plot(final[neuron, :], "--", label="Pred")
	plt.legend(shadow=True, fancybox=True)
	#plt.plot(res["x_pred"][neuron, :], label="Pred")
	plt.show()





# data_convolve = np.zeros((data.shape[0], data.shape[1] + 2 * radius))
#
# for neuron in range(data.shape[0]):
# 	data[neuron, :] = data[neuron, :] - np.min(data[neuron, :])
# 	data[neuron, :] = data[neuron, :] / np.max(data[neuron, :])
# 	data_convolve[neuron, :] = scipy.signal.convolve(data[neuron, :], gauss)
# 	data_convolve[neuron, :] /= np.max(data_convolve[neuron, :])
#
# deconvolve_data = np.zeros_like(data)
#
# for neuron in range(data.shape[0]):
# 	deconv, _ = scipy.signal.deconvolve(data_convolve[neuron, :], gauss)
# 	deconvolve_data[neuron, :] = deconv
#
#
# for neuron in range(deconvolve_data.shape[0]):
# 	plt.plot(data_convolve[neuron, :], alpha=0.5)
# 	#plt.plot(deconvolve_data[neuron, :], linestyle="--")
# 	plt.show()

forward_weights = random_matrix(200, 0.2)


# res = train_with_params(
# 	true_time_series=data,
# 	learning_rate=1e-2,
# 	epochs=300,
# 	forward_weights=forward_weights,
# 	std_weights=1,
# 	dt=0.02,
# 	mu=0.0,
# 	mean_mu=0,
# 	std_mu=1,
# 	r=0.1,
# 	mean_r=0.2,
# 	std_r=0,
# 	tau=0.1,
# 	learn_mu=True,
# 	learn_r=True,
# 	learn_tau=True,
# 	device=torch.device("cpu")
# )