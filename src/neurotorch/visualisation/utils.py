from matplotlib import pyplot as plt

from .time_series_visualisation import Visualise, VisualiseKMeans, VisualisePCA, VisualiseUMAP


def UMAP_PCA_report(pred_viz: Visualise, target_viz: Visualise, **kwargs):
	fig, axes = plt.subplots(ncols=2, nrows=3, figsize=kwargs.get("figsize", (16, 8)))
	Visualise.number_axes(axes)
	viz_pca_target = Visualise(
		timeseries=target_viz.timeseries,
		shape=target_viz.shape,
	)
	viz_pca = VisualisePCA(
		timeseries=pred_viz.timeseries,
		shape=pred_viz.shape,
	).trajectory_pca(target=viz_pca_target, fig=fig, axes=axes[:, 1], show=False)
	viz_umap_target = Visualise(
		timeseries=target_viz.timeseries,
		shape=target_viz.shape,
	)
	viz_umap = VisualiseUMAP(
		timeseries=pred_viz.timeseries,
		shape=pred_viz.shape,
	).trajectory_umap(
		UMAPs=(1, 2), target=viz_umap_target,
		fig=fig, axes=axes[:, 0],
		filename=kwargs.get("filename", None),
		show=kwargs.get("show", True)
	)
