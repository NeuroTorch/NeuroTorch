import os
from typing import Iterable, Optional, Tuple, Any

import numpy as np
import torch
from matplotlib import pyplot as plt


@torch.no_grad()
def visualize_init_final_weights(
        initial_wights: Any,
        final_weights: Any,
        filename: Optional[str] = None,
        show: bool = False,
        fig: Optional[plt.Figure] = None,
        axes: Optional[Iterable[plt.Axes]] = None,
        **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    import neurotorch as nt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib import colors
    initial_wights, final_weights = nt.to_numpy(initial_wights), nt.to_numpy(final_weights)

    assert (fig is None) == (axes is None), "If fig is None, axes must be None"
    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    else:
        assert isinstance(axes, Iterable), "If fig is not None, axes must be an iterable"
    axes_view = np.asarray(axes).ravel()

    if kwargs.get("cmp_zero_center", True):
        min_value = min(np.min(initial_wights), np.min(final_weights))
        max_value = max(np.max(initial_wights), np.max(final_weights))
        if not np.isclose(min_value, max_value) and min_value < max_value and min_value < 0 < max_value:
            divnorm = colors.TwoSlopeNorm(
                vmin=min_value,
                vcenter=0.0,
                vmax=max_value
            )
            im = axes_view[0].imshow(initial_wights, cmap="RdBu_r", norm=divnorm)
        else:
            im = axes_view[0].imshow(initial_wights, cmap="RdBu_r")
    else:
        im = axes_view[0].imshow(initial_wights, cmap="RdBu_r")
    axes_view[1].imshow(final_weights, cmap=im.cmap, extent=im.get_extent())
    cax = inset_axes(
        axes_view[1],
        width="5%",
        height="100%",
        loc='lower left',
        bbox_to_anchor=(1.05, 0.0, 1.0, 1.0),
        bbox_transform=axes_view[1].transAxes,
        borderpad=0,
    )

    if kwargs.get('compute_dale', "dale_law_kwargs" in kwargs):
        dale_law_kwargs = kwargs.get('dale_law_kwargs', {})
        dale_law_kwargs.setdefault('seed', 0)
        initial_dale = nt.to_numpy(nt.DaleLaw([nt.to_tensor(initial_wights)], **dale_law_kwargs)())
        final_dale = nt.to_numpy(nt.DaleLaw([nt.to_tensor(final_weights)], **dale_law_kwargs)())
        axes_view[0].set_title(f"Initial Weights (Dale loss = {initial_dale:.3f})")
        axes_view[1].set_title(f"Final Weights (Dale loss = {final_dale:.3f})")
    else:
        axes_view[0].set_title("Initial Weights")
        axes_view[1].set_title("Final Weights")

    fig.colorbar(im, cax=cax)
    fig.set_tight_layout(False)
    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename)
    if show:
        plt.show()
    return fig, axes


