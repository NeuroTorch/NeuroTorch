from matplotlib import pyplot as plt

from .time_series_visualisation import Visualise, VisualiseKMeans, VisualisePCA, VisualiseUMAP
from ..metrics.losses import PVarianceLoss


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


def mix_report(pred_viz: Visualise, target_viz: Visualise, **kwargs):
    fig, axes = plt.subplots(ncols=2, nrows=3, figsize=kwargs.get("figsize", (16, 8)))
    Visualise.number_axes(axes)
    pVar = PVarianceLoss()(pred_viz.timeseries, target_viz.timeseries)
    title = f"Network Prediction (pVar: {pVar:.3f})"
    fig.suptitle(title, fontsize=kwargs.get("fontsize", 16))

    pred_viz.plot_timeseries_comparison(
        target_viz.timeseries, kwargs.get("spikes"),
        n_spikes_steps=kwargs.get("n_spikes_steps"),
        title="", desc=kwargs.get("desc", "Prediction"),
        fig=fig, axes=axes[:, 0],
        traces_to_show=kwargs.get("traces_to_show", ["best", "most_var", "typical_0"]),
        traces_to_show_names=kwargs.get("traces_to_show_names", [
            "Best Neuron Prediction", "Most variable Neuron Prediction", "Typical Neuron Prediction"
        ]),
        show=False,
    )
    space = kwargs.get("space", "UMAP")
    if isinstance(pred_viz, (VisualiseUMAP, VisualisePCA)):
        pred_viz_space = pred_viz
    elif space == "UMAP":
        pred_viz_space = VisualiseUMAP(
            timeseries=pred_viz.timeseries,
            shape=pred_viz.shape,
        )
    elif space == "PCA":
        pred_viz_space = VisualisePCA(
            timeseries=pred_viz.timeseries,
            shape=pred_viz.shape,
        )
    else:
        raise ValueError(f"Unknown space: {space}. Try 'UMAP' or 'PCA'.")
    pred_viz_space.trajectory_umap(
        UMAPs=(1, 2), target=target_viz,
        fig=fig, axes=axes[:, 1],
        filename=kwargs.get("filename", None),
        show=kwargs.get("show", True)
    )



