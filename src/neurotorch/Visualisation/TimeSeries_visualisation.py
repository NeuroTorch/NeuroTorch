import abc
import numpy as np
from applications.FakeTimeSeries_forecasting_WilsonCowan.dataset import WilsonCowanTimeSeries
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from scipy import interpolate
from umap import UMAP


class Visualise:
    """
    This class is used to visualise the time series without any form of clustering.
    The time series can be visualised in the following ways:
        1. Animate the time series
        2. Heatmap of the time series
        3. Rigid plot of the time series
        4. Plot all the neuronal activity in one figure
    """
    def __init__(self,
                 timeseries: np.array,
                 apply_zscore: bool = False
                 ):
        """
        :param timeseries: Time series of shape (num_sample, num_variable). Make sure the time series is a numpy array
        """
        self.timeseries = timeseries
        self.num_sample, self.num_variable = self.timeseries.shape
        if apply_zscore:
            for i in range(self.num_sample):
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
        num_frames = self.num_variable // step
        connectome = nx.from_numpy_array(forward_weights)
        pos = nx.spring_layout(connectome)
        fig, ax = plt.subplots(figsize=(7, 7))
        nx.draw_networkx_nodes(connectome, pos, ax=ax, node_size=node_size, node_color=self.timeseries[:, 0],
                               cmap="hot")
        nx.draw_networkx_edges(connectome, pos, ax=ax, width=1.0, alpha=alpha)
        x, y = ax.get_xlim()[0], ax.get_ylim()[1]
        plt.axis("off")
        text = ax.text(0, 1.15, rf"$t = 0 / {self.num_variable * dt}$", ha="center")
        plt.tight_layout(pad=0)

        def _animation(i):
            nodes = nx.draw_networkx_nodes(connectome, pos, ax=ax, node_size=node_size,
                                           node_color=self.timeseries[:, i * step], cmap="hot")
            text.set_text(rf"$t = {i * step * dt:.3f} / {self.num_variable * dt}$")
            return nodes, text

        anim = animation.FuncAnimation(fig, _animation, frames=num_frames, interval=time_interval, blit=True)
        plt.show()

    def plot_timeseries(self, dt: float = None):
        """
        Plot all the neuronal activity in one figure.
        :param dt: Time step
        """
        if dt is not None:
            time = np.linspace(0, self.num_variable * dt, self.num_variable)
            plt.xlabel("Time [s]")
        else:
            time = np.linspace(0, self.num_variable, self.num_variable)
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
        for i in range(self.num_sample):
            shifted_timeseries = self.timeseries[i] - np.min(self.timeseries[i])
            shifted_timeseries = shifted_timeseries / np.max(shifted_timeseries) + 0.8 * i
            ax.plot(shifted_timeseries, c="k", alpha=0.9, linewidth=1.0)
        plt.show()


class VisualiseKMeans(Visualise):
    """
    Visualise the time series using only a K-means algorithm of clustering
    """
    def __init__(self,
                 timeseries: np.array,
                 apply_zscore: bool = False,
                 n_clusters: int = 13,
                 random_state: int = 0
                 ):
        """
        :param timeseries: Time series of shape (num_sample, num_sample). Make sure the time series is a numpy array
        :param apply_zscore: Whether to apply z-score or not.
        :param n_clusters: Number of clusters.
        :param random_state: Determines random number generation for centroid initialization.
        Use an int to make the randomness deterministic
        """
        super().__init__(
            timeseries=timeseries,
            apply_zscore=apply_zscore
        )
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.timeseries = self._permute_timeseries()

    def _compute_kmeans_labels(self):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(self.timeseries)
        return kmeans.labels_

    def _permute_timeseries(self):
        labels = self._compute_kmeans_labels()
        cluster_labels = np.unique(labels)
        permuted_timeseries = np.zeros_like(self.timeseries)
        position = 0
        for cluster in cluster_labels:
            for i in range(self.num_sample):
                if labels[i] == cluster:
                    permuted_timeseries[position] = self.timeseries[i]
                    position += 1
        print(permuted_timeseries)
        return permuted_timeseries


class VisualisePCA(Visualise):
    """
    Visualise the time series using PCA algorithm of dimensionality reduction. PCA is apply on the variable
    """
    def __init__(self, timeseries: np.array,
                 apply_zscore: bool = False,
                 n_PC: int = 5
                 ):
        """
        :param timeseries: Time series of shape (num_sample, num_sample). Make sure the time series is a numpy array
        :param apply_zscore: Whether to apply z-score or not.
        :param n_PC: Number of principal components.
        """
        super().__init__(
            timeseries=timeseries,
            apply_zscore=apply_zscore
        )
        self.n_PC = n_PC
        self.params = {}
        self.reduced_timeseries, self.params["var_ratio"], self.params["var_ratio_cumsum"] = self._compute_pca(n_PC)
        self.kmean_label = None

    def _compute_pca(self, n_PC: int):
        """
        Compute PCA of the time series.
        :return: timeseries in PCA space, variance ratio, variance ratio cumulative sum
        """
        pca = PCA(n_components=n_PC).fit(self.timeseries)
        reduced_timeseries = pca.transform(self.timeseries)
        return reduced_timeseries, pca.explained_variance_ratio_, pca.explained_variance_ratio_.cumsum()

    def with_kmeans(self, n_clusters: int = 13, random_state: int = 0):
        """
        Apply K-means clustering to the PCA space as coloring.
        :param n_clusters: Number of clusters.
        :param random_state: Determines random number generation for centroid initialization.
            Example: VisualisePCA(data).with_kmeans(n_clusters=13, random_state=0).scatter_pca()
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(self.reduced_timeseries)
        self.kmean_label = kmeans.labels_
        return self

    def scatter_pca(self, PCs : list = [1, 2], color_sample: bool = False):
        """
        Plot the scatter plot of the PCA space in 2D or 3D.
        :param PCs: List of PCs to plot. Always a list of length 2.
        :param color_sample: Whether to color the sample or not.
        """
        dimension = len(PCs)
        if dimension < 2 or dimension > 3:
            raise ValueError("PCs must be a list of 2 or 3 elements. Can only plot PCs in 2D or 3D")
        if self.kmean_label is not None and color_sample:
            raise ValueError("You can only apply color based on k-mean or the sample, not both")
        if max(PCs) > self.n_PC:
            raise ValueError("PCs must be less than or equal to the number of PC")
        color = None
        if self.kmean_label is not None:
            color = self.kmean_label
        if color_sample:
            color = range(self.num_sample)

        if dimension == 2:
            plt.title("Two-dimensional PCA embedding")
            plt.xlabel(f"PC {PCs[0]}")
            plt.ylabel(f"PC {PCs[1]}")
            plt.scatter(self.reduced_timeseries[:, PCs[0] - 1], self.reduced_timeseries[:, PCs[1] - 1],
                       c=color, cmap="RdBu_r")
            if self.kmean_label is not None or color_sample:
                plt.colorbar()
        if dimension == 3:
            fig, ax = plt.subplots(figsize=(16, 8))
            ax = fig.add_subplot(projection="3d")
            ax.set_title("Three-dimensional PCA embedding")
            ax.set_xlabel(f"PC {PCs[0]}")
            ax.set_ylabel(f"PC {PCs[1]}")
            ax.set_zlabel(f"PC {PCs[2]}")
            ax.scatter(self.reduced_timeseries[:, PCs[0] - 1], self.reduced_timeseries[:, PCs[1] - 1],
                       self.reduced_timeseries[:, PCs[2] - 1], c=color, cmap="RdBu_r")
        plt.show()

    def trajectory_pca(self, PCs: list = [1, 2], with_smooth: bool = True, degree: int = 5, condition: float = 5,
                       reduction: int = 1):
        """
        Plot the trajectory of the PCA space in 2D.
        :param PCs: List of PCs to plot. Always a list of length 2.
        :param with_smooth: Whether to smooth the trajectory or not.
        :param degree: Degree of the polynomial used for smoothing.
        :param condition: Smoothing condition.
        :param reduction: Number by which we divide the number of samples.
        """
        if len(PCs) != 2:
            raise ValueError("Can only plot the trajectory in PCA space in 2D. PCs must have a length of 2")
        if max(PCs) > self.n_PC:
            raise ValueError("PCs must be less than or equal to the number of PC")
        plt.figure(figsize=(16, 8))
        x = self.reduced_timeseries[:, PCs[0] - 1]
        x = x[::reduction]
        y = self.reduced_timeseries[:, PCs[1] - 1]
        y = y[::reduction]
        if with_smooth:
            smoothed_timeseries = interpolate.splprep([x, y], s=condition, k=degree, per=False)[0]
            x, y = interpolate.splev(np.linspace(0, 1, 1000), smoothed_timeseries)
        plt.plot(x, y)
        plt.title("Two-dimensional trajectory in PCA space")
        if with_smooth:
            plt.title("Two-dimensional trajectory in PCA space with smoothing")
        plt.xlabel(f"PC {PCs[0]}")
        plt.ylabel(f"PC {PCs[1]}")
        plt.show()


class VisualiseUMAP(Visualise):

    def __init__(self, timeseries: np.array,
                 apply_zscore: bool = False,
                 n_neighbors: int = 10,
                 min_dist: float = 0.5,
                 n_components: int = 2
                 ):
        super().__init__(
            timeseries=timeseries,
            apply_zscore=apply_zscore
        )
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.reduced_timeseries = self._compute_umap()

    def _compute_umap(self):
        umap = UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=self.n_components,
            metric="euclidian"
        )
        reduced_timeseries = umap.fit_transform(self.timeseries)
        return reduced_timeseries

    def with_kmeans(self, n_clusters: int = 13, random_state: int = 0):
        """
        Apply K-means clustering to the PCA space as coloring.
        :param n_clusters: Number of clusters.
        :param random_state: Determines random number generation for centroid initialization.
            Example: VisualisePCA(data).with_kmeans(n_clusters=13, random_state=0).scatter_umap()
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(self.reduced_timeseries)
        self.kmean_label = kmeans.labels_
        return self

    def scatter_umap(self):
        pass

    def trajectory_umap(self):
        pass


if __name__ == '__main__':
    # # large dataset
    # ts = np.load('timeSeries_2020_12_16_cr3_df.npy')
    # n_neurons, n_steps = ts.shape
    # # z-scored data
    # ts_z = np.zeros((n_neurons, n_steps))
    # for i in range(n_neurons):
    #     ts_z[i, :] = (ts[i, :] - np.mean(ts[i, :])) / np.std(ts[i, :])
    # # small sample
    # sample_size = 500
    # sample = np.random.randint(n_neurons, size=sample_size)
    # data = ts_z[sample, :]
    # VisualisePCA(data.T).trajectory_pca([1, 2], with_smooth=True, reduction=5)


    i = 350  # Num of neurons
    num_step = 5000
    dt = 0.1
    t_0 = np.random.rand(i, )
    forward_weights = 8 * np.random.randn(i, i)
    mu = 0
    r = np.random.rand(i, ) * 2
    tau = 1

    dynamic = WilsonCowanTimeSeries(num_step, dt, t_0, forward_weights, mu, r, tau)
    VisualisePCA(dynamic.compute_timeseries()).animate(forward_weights, dt, time_interval=0.1)
