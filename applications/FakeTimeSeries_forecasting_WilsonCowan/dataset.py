from typing import Callable, Optional, Tuple

import numpy
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
from matplotlib import animation
from torchvision.transforms import ToTensor
import networkx as nx


class WilsonCowanTimeSeries:
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
            plt.imshow(self.forward_weights, cmap="RdBu_r")
            plt.colorbar()
            plt.show()
        timeseries = self.compute_timeseries()
        time = np.linspace(0, self.num_step * self.dt, self.num_step)
        plt.plot(time.T, timeseries.T)
        plt.xlabel('Time')
        plt.ylabel('Neuronal activity')
        plt.ylim([0, 1])
        plt.show()

    # TODO : add documentation and adapt variable's name
    def animate_timeseries(self, step=4, time_interval=1, node_size=50, alpha=0.01):
        """
        Animate the time series. The position of the nodes are obtained using the spring layout.
        Spring-Layout use the Fruchterman-Reingold force-directed algorithm. For more information,
        please refer to the following documentation of networkx:
        https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html

        :param step: Number of time step between two animation frames.
            example: if step = 4, the animation will play at t = 0, t = 4, t = 8, t = 12 ...
        :param time_interval: Time interval between two animation frames (in milliseconds)
        :param node_size: Size of the nodes
        :param alpha: Density of the connections. Small network should have a higher alpha value.
        """
        num_frames = int(self.num_step / step)
        timeseries = self.compute_timeseries()
        connectome = nx.from_numpy_array(self.forward_weights)
        pos = nx.spring_layout(connectome)
        fig, ax = plt.subplots(figsize=(7, 7))
        nx.draw_networkx_nodes(connectome, pos, ax=ax, node_size=node_size, node_color=timeseries[:, 0], cmap="hot")
        nx.draw_networkx_edges(connectome, pos, ax=ax, width=1.0, alpha=0.01)
        x, y = ax.get_xlim()[0], ax.get_ylim()[1]
        plt.axis("off")
        text = ax.text(0, 1.15, rf"$t = 0 / {self.num_step * self.dt}$", ha="center")
        plt.tight_layout(pad=0)

        def _animate(i):
            nodes = nx.draw_networkx_nodes(connectome, pos, ax=ax, node_size=node_size,
                                           node_color=timeseries[:, i * step], cmap="hot")
            text.set_text(rf"$t = {i * step * self.dt:.3f} / {self.num_step * self.dt}$")
            return nodes, text

        anim = animation.FuncAnimation(fig, _animate, frames=num_frames, interval=time_interval, blit=True)
        plt.show()



# Example
i = 300  # Num of neurons
num_step = 5000
dt = 0.1
t_0 = np.random.rand(i,)
forward_weights = 8 * np.random.randn(i, i)
mu = 0
r = np.random.rand(i,) * 2
tau = 1

dynamic = WilsonCowanTimeSeries(num_step, dt, t_0, forward_weights, mu, r, tau)

#dynamic.plot_timeseries(show_matrix=True)
#dynamic.animate_timeseries(time_interval=0.1)



# TODO : Create dataset function for training and do documentation

class WilsonCowanDataset(Dataset):
    def __init__(self,
                TimeSeries: torch.Tensor or numpy.array,
                 *,
                transform: Optional[Callable] = None,
    ):
        super().__init__()


    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


