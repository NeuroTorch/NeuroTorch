# File for test. Will not be added to the repository.

import numpy as np
from dataset import WilsonCowanTimeSeries

# Example
i = 300  # Num of neurons
num_step = 5000
dt = 0.1
t_0 = np.random.rand(i,)

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
    return W


forward_weights = 40 * random_matrix(i, 0.2)
mu = 0
r = 0
tau = 1

dynamic = WilsonCowanTimeSeries(num_step, dt, t_0, forward_weights, mu, r, tau)
dynamic.plot_timeseries(show_matrix=True)
dynamic.animate_timeseries(time_interval=0.1)

