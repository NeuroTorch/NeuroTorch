import abc
import numpy as np
from applications.FakeTimeSeries_forecasting_WilsonCowan.dataset import WilsonCowanTimeSeries
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN


class Visualise:
    """
    This class is used to visualise the time series without any form of clustering.
    The time series can be visualised in the following ways:
        1. Animate the time series
        2. Heatmap of the time series
        3. Rigid plot of the time series
        4. Plot all the neuronal activity in one figure
    """
    def __init__(self, timeseries: np.array, apply_zscore: bool = False):
        """
        :param timeseries: Time series of shape (num_neuron, num_step). Make sure the time series is a numpy array
        """
        self.timeseries = timeseries
        self.num_neuron, self.num_step = timeseries.shape
        if apply_zscore:
            for i in range(self.num_neuron):
                self.timeseries[i] = (self.timeseries[i] - np.mean(self.timeseries[i])) / np.std(self.timeseries[i])

    def animate(self, forward_weights, dt, step: int = 4, time_interval: float = 1.0,
                node_size: float = 50, alpha: float = 0.01):
        """
        Animate the time series. The position of the nodes are obtained using the spring layout.
        Spring-Layout use the Fruchterman-Reingold force-directed algorithm. For more information,
        please refer to the following documentation of networkx:
        https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html

        :param forward_weights: Weight matrix of size (number of neurons, number of neurons).
        :param dt: Time step.
        :param step: Number of time step between two animation frames.
            example: if step = 4, the animation will play at t = 0, t = 4, t = 8, t = 12 ...
        :param time_interval: Time interval between two animation frames (in milliseconds)
        :param node_size: Size of the nodes
        :param alpha: Density of the connections. Small network should have a higher alpha value.
        """
        num_frames = self.num_step // step
        connectome = nx.from_numpy_array(forward_weights)
        pos = nx.spring_layout(connectome)
        fig, ax = plt.subplots(figsize=(7, 7))
        nx.draw_networkx_nodes(connectome, pos, ax=ax, node_size=node_size, node_color=self.timeseries[:, 0],
                               cmap="hot")
        nx.draw_networkx_edges(connectome, pos, ax=ax, width=1.0, alpha=alpha)
        x, y = ax.get_xlim()[0], ax.get_ylim()[1]
        plt.axis("off")
        text = ax.text(0, 1.15, rf"$t = 0 / {self.num_step * dt}$", ha="center")
        plt.tight_layout(pad=0)

        def _animation(i):
            nodes = nx.draw_networkx_nodes(connectome, pos, ax=ax, node_size=node_size,
                                           node_color=self.timeseries[:, i * step], cmap="hot")
            text.set_text(rf"$t = {i * step * dt:.3f} / {self.num_step * dt}$")
            return nodes, text

        anim = animation.FuncAnimation(fig, _animation, frames=num_frames, interval=time_interval, blit=True)
        plt.show()

    def plot_timeseries(self, dt: float = None):
        """
        Plot all the neuronal activity in one figure.
        :param dt: Time step
        """
        if dt is not None:
            time = np.linspace(0, self.num_step * dt, self.num_step)
            plt.xlabel("Time [s]")
        else:
            time = np.linspace(0, self.num_step, self.num_step)
            plt.xlabel("Time step [-]")

        plt.plot(time.T, self.timeseries.T)
        plt.ylabel("Neuron activity [-]")
        plt.show()

    def heatmap(self, show_axis: bool = True, interpolation: str = "nearest", cmap: str = "RdBu_r",
                v: list[float, float] = [0.0, 1.0]):
        """
        Plot the heatmap of the time series.
        :param show_axis: Whether to show the axis or not.
        :param interpolation: Type of interpolation between the time step.
        :param cmap: Colormap of the heatmap.
        :param v: Range of the colorbar.
        """
        plt.figure(figsize=(16, 8))
        plt.imshow(self.timeseries, interpolation=interpolation, aspect="auto", cmap=cmap, vmin=v[0], vmax=v[1])
        plt.xlabel("Time step [-]")
        plt.ylabel("Neuron ID [-]")
        if not show_axis:
            plt.axis("off")
        plt.colorbar()
        plt.show()

    def rigidplot(self, show_axis: bool = False):
        """
        Plot the rigid plot of the time series.
        :param show_axis: Whether to show the axis or not.
        """
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_xlabel("Time step [-]")
        ax.set_ylabel("Neuron ID [-]")
        if not show_axis:
            ax.axis("off")
        for i in range(self.num_neuron):
            shifted_timeseries = self.timeseries[i] - np.min(self.timeseries[i])
            shifted_timeseries = shifted_timeseries / np.max(shifted_timeseries) + 0.8 * i
            ax.plot(shifted_timeseries, c="k", alpha=0.9, linewidth=1.0)
        plt.show()


class VisualisePCA(Visualise):

    def __init__(self, timeseries: np.array, apply_zscore: bool = False):
        super().__init__(
            timeseries=timeseries,
            apply_zscore=apply_zscore
        )

class VisualiseKMeans(Visualise):

    def __init__(self, timeseries: np.array, apply_zscore: bool = False, n_cluster: int = 13, random_state: int = 0):
        super().__init__(
            timeseries=timeseries,
            apply_zscore=apply_zscore
        )
        self.n_cluster = n_cluster
        self.random_state = random_state
        self.timeseries = self._permute_timeseries()

    def _compute_kmeans_labels(self):
        kmeans = KMeans(n_clusters=self.n_cluster, random_state=self.random_state).fit(self.timeseries)
        return kmeans.labels_

    def _permute_timeseries(self):
        labels = self._compute_kmeans_labels()
        cluster_labels = np.unique(labels)
        permuted_timeseries = np.zeros_like(self.timeseries)
        position = 0
        for cluster in cluster_labels:
            for i in range(self.num_neuron):
                if labels[i] == cluster:
                    permuted_timeseries[position] = self.timeseries[i]
                    position += 1
        return permuted_timeseries

    def _permutedV2(self):
        pass

    def heatmap(self, show_axis: bool = True, interpolation: str = "nearest", cmap: str = "RdBu_r",
                v: list[float, float] = [0.0, 1.0]):
        """
        Plot the heatmap of the time series.
        :param show_axis: Whether to show the axis or not.
        :param interpolation: Type of interpolation between the time step.
        :param cmap: Colormap of the heatmap.
        :param v: Range of the colorbar.
        """
        plt.figure(figsize=(16, 8))
        plt.imshow(self.timeseries, interpolation=interpolation, aspect="auto", cmap=cmap, vmin=v[0], vmax=v[1])
        plt.xlabel("Time step [-]")
        plt.ylabel("Neuron ID [-]")
        if not show_axis:
            plt.axis("off")
        plt.colorbar()
        plt.show()




if __name__ == '__main__':
    # large dataset
    ts = np.load('timeSeries_2020_12_16_cr3_df.npy')
    n_neurons, n_steps = ts.shape
    # z-scored data
    ts_z = np.zeros((n_neurons, n_steps))
    for i in range(n_neurons):
        ts_z[i, :] = (ts[i, :] - np.mean(ts[i, :])) / np.std(ts[i, :])
    # small sample
    sample_size = 350
    sample = np.random.randint(n_neurons, size=sample_size)
    data = ts_z[sample, :]
    print(data.shape)
    # Visualise(ts, apply_zscore=True).heatmap(v=[-3, 3], show_axis=False)
    VisualiseKMeans(data, apply_zscore=False, n_cluster=13, random_state=0).heatmap(v=[-3, 3], show_axis=False)

    # i = 200  # Num of neurons
    # num_step = 5000
    # dt = 0.1
    # t_0 = np.random.rand(i, )
    # forward_weights = 8 * np.random.randn(i, i)
    # mu = 0
    # r = np.random.rand(i, ) * 2
    # tau = 1
    #
    # dynamic = WilsonCowanTimeSeries(num_step, dt, t_0, forward_weights, mu, r, tau)