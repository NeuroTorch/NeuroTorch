import os
from copy import deepcopy
from typing import Optional, Tuple, Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import umap
from matplotlib import animation
from scipy import interpolate
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

from ..dimension import Dimension, DimensionProperty, Size, DimensionLike, DimensionsLike
from ..metrics import PVarianceLoss
from ..transforms.base import to_numpy, to_tensor


class Visualise:
	"""
	This class is used to visualise the time series without any form of clustering.
	The time series can be visualised in the following ways:
		1. Animate the time series
		2. Heatmap of the time series
		3. Rigid plot of the time series
		4. Plot all the neuronal activity in one figure
	Further visualisation are already added. You can visualise the time series with clustering methods. You can
	then visualise those clustered time series in scatter/trajectory in their clustered space.
	"""

	def __init__(
			self,
			timeseries: Any,
			shape: Optional[DimensionsLike] = None,
			apply_zscore: bool = False
		):
		"""
		:param timeseries: Time series of shape (Time Steps, Features).
		:param shape: Shape of the time series. If shape is None, the shape is inferred from the time series. Useful
		to identify the dtype of the time series. If the shape is Size, make sure to set the name of the dimensions as
		you want them to be displayed.
		"""
		self.timeseries = to_numpy(timeseries)
		self._given_timeseries = deepcopy(self.timeseries)
		self.shape = self._set_dimension(shape)
		if apply_zscore:
			self._zscore_timeseries()
	
	def _zscore_timeseries(self):
		"""
		Z-score the time series.
		"""
		for i in range(int(self.shape[-1])):
			self.timeseries[:, i] = (
					(self.timeseries[:, i] - np.mean(self.timeseries[:, i])) / np.std(self.timeseries[:, i])
			)

	def _set_dimension(self, shape: Optional[DimensionsLike]) -> Size:
		"""
		Identify the shape of the time series. Will transpose the time series to have the time dimension as the first
		dimension.
		
		:param shape: Shape of the time series. Use object Size given in this package.
		:return: The shape of the time series.
		"""
		if shape is None:
			shape = self.timeseries.shape

		shape = Size(shape)
		assert len(shape) == 2, "The shape of the time series must be 2 dimensional"
		if all(dim.dtype == DimensionProperty.NONE for dim in shape):
			shape[0].dtype = DimensionProperty.TIME
			for dim in shape:
				if dim.dtype == DimensionProperty.TIME:
					dim.name = "Time Steps"
				elif dim.dtype == DimensionProperty.NONE:
					dim.name = "Features"
		for i, dim in enumerate(shape):
			if dim.size is None:
				dim.size = self.timeseries.shape[i]
		
		if shape[0].dtype == DimensionProperty.NONE:
			self.timeseries = self.timeseries.T
			self.shape = Size(shape.dimensions[::-1])
		return shape

	def animate(
			self,
			forward_weights: np.ndarray,
			dt: float = 1.0,
			step: int = 1,
			time_interval: float = 1.0,
			node_size: float = 50,
			alpha: float = 0.01,
			filename: Optional[str] = None,
			file_extension: Optional[str] = None,
			show: bool = False,
			**kwargs
	):
		"""
		Animate the time series. The position of the nodes are obtained using the spring layout.
		Spring-Layout use the Fruchterman-Reingold force-directed algorithm. For more information,
		please refer to the following documentation of networkx:
		https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html

		:param forward_weights: Weight matrix of size (number of neurons, number of neurons).
		:param dt: Time step between two time steps.
		:param step: Number of time step between two animation frames.
			example: if step = 4, the animation will play at t = 0, t = 4, t = 8, t = 12 ...
		:param time_interval: Time interval between two animation frames (in milliseconds)
		:param node_size: Size of the nodes
		:param alpha: Density of the connections. Small network should have a higher alpha value.
		:param filename: Name of the file to save the animation. If filename is None, the animation will not be saved.
		The application imagemagick is required to save the animation.
		:param file_extension: Extension of the file to save the animation. The available extensions are: mp4 and gif.
		:param show: If True, the animation will be displayed.
		:param kwargs: Keyword arguments.
		
		:keyword fps: Frames per second. Default is 30.
		"""
		num_frames = int(self.shape[0]) // step
		connectome = nx.from_numpy_array(forward_weights)
		pos = nx.spring_layout(connectome)
		fig, ax = plt.subplots(figsize=(7, 7))
		nx.draw_networkx_nodes(
			connectome,
			pos,
			ax=ax,
			node_size=node_size,
			node_color=self.timeseries[0],
			cmap="hot"
		)
		nx.draw_networkx_edges(connectome, pos, ax=ax, width=1.0, alpha=alpha)
		x, y = ax.get_xlim()[0], ax.get_ylim()[1]
		plt.axis("off")
		text = ax.text(0, 1.08, rf"$t = 0 / {int(self.shape[0]) * dt}$", ha="center")
		plt.tight_layout(pad=0)

		def _animation(i):
			nodes = nx.draw_networkx_nodes(
				connectome,
				pos,
				ax=ax, node_size=node_size,
				node_color=self.timeseries[i * step],
				cmap="hot"
			)
			text.set_text(rf"$t = {i * step * dt:.3f} / {int(self.shape[0]) * dt}$")
			return nodes, text

		anim = animation.FuncAnimation(fig, _animation, frames=num_frames, interval=time_interval, blit=True)
		if filename is not None:
			os.makedirs(os.path.dirname(filename), exist_ok=True)
			if file_extension is None:
				if '.' in filename:
					file_extension = filename.split('.')[-1]
				else:
					file_extension = 'gif'
			assert file_extension in ["mp4", "gif"], "The extension of the file must be mp4 or gif."
			if filename.endswith(file_extension):
				filename = ''.join(filename.split('.')[:-1])
			anim.save(f"{filename}.{file_extension}", writer="imagemagick", fps=kwargs.get("fps", 30))
		if show:
			plt.show()

	def plot_timeseries(
			self,
			filename: Optional[str] = None,
			show: bool = False,
			**kwargs
	):
		"""
		Plot all the neuronal activity in one figure.
		
		:param filename: Name of the file to save the figure. If filename is None, the figure will not be saved.
		:param show: If True, the figure will be displayed.
		:param kwargs: Keyword arguments.
		
		:keyword figsize: Size of the figure. Default is (12, 8).
		:keyword dpi: DPI of the figure. Default is 300.
		"""
		fig, ax = plt.subplots(figsize=kwargs.get("figsize", (12, 8)))
		ax.set_xlabel(self.shape[0].name)
		ax.plot(self.timeseries)
		ax.set_ylabel(self.shape[1].name)
		if filename is not None:
			os.makedirs(os.path.dirname(filename), exist_ok=True)
			fig.savefig(filename, dpi=kwargs.get("dpi", 300))
		if show:
			plt.show()
		plt.close(fig)

	def heatmap(
			self,
			show_axis: bool = True,
			interpolation: str = "nearest",
			cmap: str = "RdBu_r",
			v: Tuple[float, float] = (0.0, 1.0),
			filename: Optional[str] = None,
			show: bool = False,
			**kwargs
	):
		"""
		Plot the heatmap of the time series.
		
		:param show_axis: Whether to show the axis or not.
		:param interpolation: Type of interpolation between the time step.
		:param cmap: Colormap of the heatmap.
		:param v: Range of the colorbar.
		:param filename: Name of the file to save the figure. If filename is None, the figure will not be saved.
		:param show: If True, the figure will be displayed.
		:param kwargs: Keyword arguments.
		
		:keyword figsize: Size of the figure. Default is (12, 8).
		:keyword dpi: DPI of the figure. Default is 300.
		"""
		fig, ax = plt.subplots(figsize=kwargs.get("figsize", (12, 8)))
		ax.set_xlabel(self.shape[0].name)
		ax.set_ylabel(self.shape[1].name)
		im = ax.imshow(self.timeseries.T, interpolation=interpolation, aspect="auto", cmap=cmap, vmin=v[0], vmax=v[1])
		if not show_axis:
			ax.axis("off")
		fig.colorbar(im)
		if filename is not None:
			os.makedirs(os.path.dirname(filename), exist_ok=True)
			fig.savefig(filename, dpi=kwargs.get("dpi", 300))
		if show:
			plt.show()
		plt.close(fig)

	def rigidplot(
			self,
			show_axis: bool = False,
			filename: Optional[str] = None,
			show: bool = False,
			**kwargs
	):
		"""
		Plot the rigid plot of the time series.
		
		:param show_axis: Whether to show the axis or not.
		:param filename: Name of the file to save the figure. If filename is None, the figure will not be saved.
		:param show: If True, the figure will be displayed.
		:param kwargs: Keyword arguments.
		
		:keyword figsize: Size of the figure. Default is (12, 8).
		:keyword dpi: DPI of the figure. Default is 300.
		"""
		fig, ax = plt.subplots(figsize=kwargs.get("figsize", (12, 8)))
		ax.set_xlabel(self.shape[0].name)
		ax.set_ylabel(self.shape[1].name)
		if not show_axis:
			ax.axis("off")
		for i in range(int(self.shape[-1])):
			shifted_timeseries = self.timeseries[:, i] - np.min(self.timeseries[:, i])
			shifted_timeseries = np.divide(
				shifted_timeseries, np.max(shifted_timeseries),
				out=np.zeros_like(shifted_timeseries),
				where=np.logical_not(np.isclose(np.max(shifted_timeseries), 0))
			) + 0.8 * i
			ax.plot(shifted_timeseries, c="k", alpha=0.9, linewidth=1.0)
		if filename is not None:
			os.makedirs(os.path.dirname(filename), exist_ok=True)
			fig.savefig(filename, dpi=kwargs.get("dpi", 300))
		if show:
			plt.show()
		plt.close(fig)
	
	def plot_single_timeseries_comparison(
			self,
			feature_index: int,
			ax: plt.Axes,
			target: Any,
			spikes: Optional[Any] = None,
			n_spikes_steps: Optional[int] = None,
			title: str = "",
			desc: str = "Prediction",
	) -> plt.Axes:
		predictions, target = to_tensor(self._given_timeseries[:, feature_index]), to_tensor(target)
		mse_loss = torch.nn.MSELoss()(predictions, target.to(predictions.device))
		pVar = 1 - mse_loss / torch.var(target.to(mse_loss.device))
		
		ax.plot(predictions.detach().cpu().numpy(), label=f"{desc} (pVar: {pVar.detach().cpu().item():.3f})")
		ax.plot(target.detach().cpu().numpy(), label="Target")
		
		if spikes is not None:
			spikes = to_tensor(spikes)
			assert len(spikes.shape) == 2, "spikes must be a 2D tensor"
			assert n_spikes_steps is not None, "n_spikes_steps must be provided if spikes is not None"
			y_max = max(target.max(), predictions.max())
			x_scatter_space = np.linspace(0, len(target), num=n_spikes_steps * len(target))
			x_scatter_spikes = []
			x_scatter_zeros = []
			for i, xs in enumerate(x_scatter_space):
				if np.isclose(
						spikes[i // n_spikes_steps][i % n_spikes_steps],
						1.0
				):
					x_scatter_spikes.append(xs)
				else:
					x_scatter_zeros.append(xs)
			ax.scatter(
				x_scatter_spikes, y=[y_max * 1.1] * len(x_scatter_spikes),
				label="Latent space", c='k', marker='|', linewidths=0.5
			)
		
		ax.set_xlabel("Time [-]")
		ax.set_ylabel("Activity [-]")
		ax.set_title(title)
		ax.legend()
		return ax
	
	def plot_timeseries_comparison(
			self,
			target: Any,
			spikes: Optional[Any] = None,
			n_spikes_steps: Optional[int] = None,
			title: str = "",
			desc: str = "Prediction",
			filename: Optional[str] = None,
			show: bool = False,
	) -> plt.Figure:
		predictions, target = to_tensor(self._given_timeseries), to_tensor(target)
		target = torch.squeeze(target.detach().cpu())
		
		errors = torch.squeeze(predictions - target.to(predictions.device))**2
		pVar = PVarianceLoss()(predictions, target.to(predictions.device))
		
		fig, axes = plt.subplots(3, 1, figsize=(15, 8))
		axes[0].plot(errors.detach().cpu().numpy())
		axes[0].set_xlabel("Time [-]")
		axes[0].set_ylabel("Squared Error [-]")
		axes[0].set_title(f"{title}, pVar: {pVar.detach().cpu().item():.3f}")
		
		mean_errors = torch.mean(errors, dim=0)
		mean_error_sort, indices = torch.sort(mean_errors)
		target = torch.squeeze(target).numpy().T
		
		best_idx, worst_idx = indices[0], indices[-1]
		if spikes is not None:
			spikes = np.squeeze(to_numpy(spikes))
			best_spikes = spikes[:, :, best_idx]
			worst_spikes = spikes[:, :, worst_idx]
		else:
			best_spikes = None
			worst_spikes = None
		self.plot_single_timeseries_comparison(
			best_idx, axes[1], target[best_idx], best_spikes,
			n_spikes_steps=n_spikes_steps,
			title=f"Best {desc}", desc=desc,
		)
		self.plot_single_timeseries_comparison(
			worst_idx, axes[2], target[worst_idx], worst_spikes,
			n_spikes_steps=n_spikes_steps,
			title=f"Worst {desc}", desc=desc,
		)
		
		fig.set_tight_layout(True)
		if filename is not None:
			os.makedirs(os.path.dirname(filename), exist_ok=True)
			fig.savefig(filename)
		if show:
			plt.show()
		plt.close(fig)
		return fig


class VisualiseKMeans(Visualise):
	"""
	Visualise the time series using only a K-means algorithm of clustering.
	"""

	def __init__(
			self,
			timeseries: Any,
			shape: Optional[DimensionsLike] = None,
			apply_zscore: bool = False,
			n_clusters: int = 13,
			random_state: int = 0
		):
		"""
		Constructor of the class.
		
		:param timeseries: Time series of shape (n_time_steps, n_neurons).
		:param shape: Shape of the time series. If shape is None, the shape is inferred from the time series. Useful
		to identify the dtype of the time series. If the shape is Size, make sure to set the name of the dimensions as
		you want them to be displayed.
		:param apply_zscore: Whether to apply z-score or not.
		:param n_clusters: Number of clusters.
		:param random_state: Determines random number generation for centroid initialization.
		Use an int to make the randomness deterministic
		"""
		super().__init__(
			timeseries=timeseries,
			shape=shape,
			apply_zscore=apply_zscore
		)
		self.n_clusters = n_clusters
		self.random_state = random_state
		self.labels = self._compute_kmeans_labels()
		self.cluster_labels = np.unique(self.labels)
		self.timeseries = self._permute_timeseries(self.timeseries)

	def _compute_kmeans_labels(self):
		kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(self.timeseries.T)
		return kmeans.labels_

	def _permute_timeseries(self, timeseries: np.ndarray):
		# TODO: optimize this method
		assert timeseries.shape[-1] == int(self.shape[-1])
		permuted_timeseries = np.zeros_like(timeseries)
		position = 0
		for cluster in self.cluster_labels:
			for i in range(int(self.shape[-1])):
				if self.labels[i] == cluster:
					permuted_timeseries[:, position] = timeseries[:, i]
					position += 1
		return permuted_timeseries


class VisualisePCA(Visualise):
	"""
	Visualise the time series using PCA algorithm of dimensionality reduction. PCA is apply on the variable.
	
	TODO: generalise the class to work with Size and to save the results in a file.
	"""

	def __init__(
			self,
			timeseries: Any,
			shape: Optional[DimensionsLike] = None,
			apply_zscore: bool = False,
			n_PC: int = 5
		):
		"""
		:param timeseries: Time series of shape (n_time_steps, n_neurons).
		:param shape: Shape of the time series. If shape is None, the shape is inferred from the time series. Useful
		to identify the dtype of the time series. If the shape is Size, make sure to set the name of the dimensions as
		you want them to be displayed.
		:param apply_zscore: Whether to apply z-score or not.
		:param n_PC: Number of principal components.
		"""
		super().__init__(
			timeseries=timeseries,
			shape=shape,
			apply_zscore=apply_zscore
		)
		self.n_PC = n_PC
		self.params = {}
		self.reduced_timeseries, self.params["var_ratio"], self.params["var_ratio_cumsum"] = self._compute_pca(n_PC)
		self.kmean_label = None

	def _compute_pca(self, n_PC: int):
		"""
		Compute PCA of the time series.
		
		:return: timeseries in PCA space, variance ratio, variance ratio cumulative sum.
		"""
		pca = PCA(n_components=n_PC).fit(self.timeseries.T)
		reduced_timeseries = pca.transform(self.timeseries.T)
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

	def scatter_pca(self, PCs: Tuple[int, ...] = (1, 2), color_sample: bool = False):
		"""
		Plot the scatter plot of the PCA space in 2D or 3D.
		
		:param PCs: List of PCs to plot. Always a list of length 2 or 3
		:param color_sample: Whether to color the sample or not.
		"""
		dimension = len(PCs)
		if dimension < 2 or dimension > 3:
			raise ValueError("PCs must be a tuple of 2 or 3 elements. Can only plot PCs in 2D or 3D")
		if self.kmean_label is not None and color_sample:
			raise ValueError("You can only apply color based on k-mean or the sample, not both")
		if max(PCs) > self.n_PC:
			raise ValueError("PCs must be less than or equal to the number of PC")
		color = None
		if self.kmean_label is not None:
			color = self.kmean_label
		if color_sample:
			color = range(int(self.shape[-1]))

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
			ax.scatter(
				self.reduced_timeseries[:, PCs[0] - 1],
				self.reduced_timeseries[:, PCs[1] - 1],
				self.reduced_timeseries[:, PCs[2] - 1],
				c=color, cmap="RdBu_r"
			)
		plt.show()

	def trajectory_pca(
			self,
			PCs: Tuple[int, int] = (1, 2),
			with_smooth: bool = True,
			degree: int = 5,
			condition: float = 5,
			reduction: int = 1
	):
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
	"""
	TODO: generalise the class to work with Size and to save the results in a file.
	"""

	def __init__(
			self,
			timeseries: Any,
			shape: Optional[DimensionsLike] = None,
			apply_zscore: bool = False,
			n_neighbors: int = 10,
			min_dist: float = 0.5,
			n_components: int = 3
		):
		super().__init__(
			timeseries=timeseries,
			shape=shape,
			apply_zscore=apply_zscore
		)
		self.n_neighbors = n_neighbors
		self.min_dist = min_dist
		self.n_components = n_components
		self.kmeans_label = None
		self.reduced_timeseries = self._compute_umap()

	def _compute_umap(self):
		fit = umap.UMAP(
			n_neighbors=self.n_neighbors,
			min_dist=self.min_dist,
			n_components=self.n_components,
			metric='euclidean'
		)
		reduced_timeseries = fit.fit_transform(self.timeseries)
		return reduced_timeseries

	def with_kmeans(self, n_clusters: int = 13, random_state: int = 0):
		"""
		Apply K-means clustering to the PCA space as coloring.
		:param n_clusters: Number of clusters.
		:param random_state: Determines random number generation for centroid initialization.
			Example: VisualisePCA(data).with_kmeans(n_clusters=13, random_state=0).scatter_umap()
		"""
		kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(self.reduced_timeseries)
		self.kmeans_label = kmeans.labels_
		return self

	def scatter_umap(self, UMAPs: Tuple[int, ...] = (1, 2), color_sample: bool = False):
		"""
		Plot the scatter plot of the UMAP space in 2D or 3D.
		:param UMAPs: List of UMAPs to plot. Always a list of length 2 or 3
		:param color_sample: Whether to color the sample or not.
		"""
		dimension = len(UMAPs)
		if dimension < 2 or dimension > 3:
			raise ValueError("UMAPs must be a tuple of 2 or 3 elements. Can only plot UMAPs in 2D or 3D")
		if self.kmeans_label is not None and color_sample:
			raise ValueError("You can only apply color based on k-mean or the sample, not both")
		if max(UMAPs) > self.n_components:
			raise ValueError("UMAPs must be less than or equal to the number of UMAP")
		color = None
		if self.kmeans_label is not None:
			color = self.kmeans_label
		if color_sample:
			color = range(self.num_sample)

		if dimension == 2:
			plt.title("Two-dimensional UMAP embedding")
			plt.xlabel(f"UMAP {UMAPs[0]}")
			plt.ylabel(f"UMAP {UMAPs[1]}")
			plt.scatter(self.reduced_timeseries[:, UMAPs[0] - 1], self.reduced_timeseries[:, UMAPs[1] - 1],
						c=color, cmap="RdBu_r")
			if self.kmeans_label is not None or color_sample:
				plt.colorbar()
		if dimension == 3:
			fig, ax = plt.subplots(figsize=(16, 8))
			ax = fig.add_subplot(projection="3d")
			ax.set_title("Three-dimensional UMAP embedding")
			ax.set_xlabel(f"UMAP {UMAPs[0]}")
			ax.set_ylabel(f"UMAP {UMAPs[1]}")
			ax.set_zlabel(f"UMAP {UMAPs[2]}")
			ax.scatter(
				self.reduced_timeseries[:, UMAPs[0] - 1],
				self.reduced_timeseries[:, UMAPs[1] - 1],
				self.reduced_timeseries[:, UMAPs[2] - 1],
				c=color, cmap="RdBu_r"
			)
		plt.show()

	def trajectory_umap(
			self,
			UMAPs: Tuple[int, int] = (1, 2),
			with_smooth: bool = True,
			degree: int = 5,
			condition: float = 5,
			reduction: int = 1
	):
		"""
		Plot the trajectory of the UMAP space in 2D.
		:param UMAPs: List of UMAPs to plot. Always a list of length 2.
		:param with_smooth: Whether to smooth the trajectory or not.
		:param degree: Degree of the polynomial used for smoothing.
		:param condition: Smoothing condition.
		:param reduction: Number by which we divide the number of samples.
		"""
		if len(UMAPs) != 2:
			raise ValueError("Can only plot the trajectory in UMAP space in 2D. UMAPs must have a length of 2")
		if max(UMAPs) > self.n_components:
			raise ValueError("UMAPs must be less than or equal to the number of UMAP")
		plt.figure(figsize=(16, 8))
		x = self.reduced_timeseries[:, UMAPs[0] - 1]
		x = x[::reduction]
		y = self.reduced_timeseries[:, UMAPs[1] - 1]
		y = y[::reduction]
		if with_smooth:
			smoothed_timeseries = interpolate.splprep([x, y], s=condition, k=degree, per=False)[0]
			x, y = interpolate.splev(np.linspace(0, 1, 1000), smoothed_timeseries)
		plt.plot(x, y)
		plt.title("Two-dimensional trajectory in UMAP space")
		if with_smooth:
			plt.title("Two-dimensional trajectory in UMAP space with smoothing")
		plt.xlabel(f"UMAP {UMAPs[0]}")
		plt.ylabel(f"UMAP {UMAPs[1]}")
		plt.show()


class VisualiseDBSCAN(Visualise):
	"""
	TODO: generalise the class to work with Size and to save the results in a file.
	"""

	def __init__(
			self,
			timeseries: Any,
			shape: Optional[DimensionsLike] = None,
			apply_zscore: bool = True,
			eps: float = 25,
			min_samples: int = 3,
			):
		super().__init__(
			timeseries=timeseries,
			shape=shape,
			apply_zscore=apply_zscore
		)
		self.eps = eps
		self.min_samples = min_samples
		self.timeseries = self._permute_timeseries()

	def _compute_dbscan(self):
		dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(self.timeseries)
		print(dbscan.labels_)
		return dbscan.labels_

	def _permute_timeseries(self):
		labels = self._compute_dbscan()
		cluster_labels = np.unique(labels)
		permuted_timeseries = np.zeros_like(self.timeseries)
		position = 0
		for cluster in cluster_labels:
			for i in range(self.num_sample):
				if labels[i] == cluster:
					permuted_timeseries[i] = self.timeseries[position]
					position += 1
		return permuted_timeseries
