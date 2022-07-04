from typing import Callable, Optional, Tuple

import numpy
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

# TODO : class that generate fake data,

class WilsonCowan_time_series:
    """
    This class can be used to generate fake data from the initial conditions and the forward weights.
    It can also be used to predict and/or plot the time series if the initial conditions and the
    forward weights are known. To have more informations about the Wilson-Cowan dynamics, please refer
    to the documentation in layers.py -> WilsonCowanLayer class.
    """
    def __init__(self,
                 num_step: int,
                 dt: float,
                 t_0: numpy.array,
                 forward_weights: numpy.array,
                 mu: numpy.array or float = 0.0,
                 r: numpy.array or float = 0.0,
                 tau: float = 1.0):
        """
        :param num_step: Number of time step in our time series
        :param dt: Time step
        :param t_0: Initial condition. array of size (number of neuron, )
        :param forward_weights: Weight matrix of size (number of neurons, number of neurons)
        :param mu: Activation threshold (number of neurons, )
        :param r: Transition rate of the RNN unit (number of neurons, )
        :param tau: Decay constant of RNN unit
        """
        self.num_step = num_step
        self.dt = dt
        self.t_0 = t_0
        self.forward_weights = forward_weights
        self.mu = mu
        self.r = r
        self.tau = tau

    @staticmethod
    def _sigmoid(x: numpy.array) -> numpy.array:
        """
        Sigmoid function
        :return: Sigmoid of x
        """
        return 1 / (1 + np.exp(-x))

    def _dydx(self, input: numpy.array) -> numpy.array:
        """
        input: array of size (number of neurons, ) -> neuronal activty at a time t
        Differential equation with format dydx = f(y, x)
        Here, we have f(t, input).
        """
        return (-input + (1 - self.r * input) * self._sigmoid(self.forward_weights @ input - self.mu)) / self.tau

    def compute_timeseries(self) -> numpy.array:
        """
        Compute a time series using Runge-Kutta of fourth order. The time series is compute
        from the initial condition t_0 and the forward weights.
        """
        num_neurons = self.t_0.shape[0]
        timeseries = np.zeros((num_neurons, self.num_step))
        timeseries[:, 0] = self.t_0
        for i in range(1, self.num_step):
            input = timeseries[:, i - 1]
            k1 = self.dt * self._dydx(input)
            k2 = self.dt * self._dydx(input + k1 / 2)
            k3 = self.dt * self._dydx(input + k2 / 2)
            k4 = self.dt * self._dydx(input + k3)
            timeseries[:, i] = input + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return timeseries

    def plot_timeseries(self, show_matrix=False):
        """
        Plot the time series.
        """
        if show_matrix:
            plt.imshow(self.forward_weights, cmap="RdBu")
            plt.show()
        timeseries = self.compute_timeseries()
        plt.plot(timeseries.T)
        plt.xlabel('Time')
        plt.ylabel('Neuronal activity')
        plt.ylim([0, 1])
        plt.show()



i = 250  # Num of neurons
num_step = 1000
dt = 0.1
t_0 = np.random.randn(i,)
forward_weights = 8 * np.random.randn(i, i)
mu = 0
r = np.random.rand(i,) * 2
tau = 1

WilsonCowan_time_series(num_step, dt, t_0, forward_weights, mu, r, tau).plot_timeseries(show_matrix=True)


