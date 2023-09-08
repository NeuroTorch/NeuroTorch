from typing import Any, Type, Union, Optional

import numpy as np
import torch

from . import ConstantValuesTransform, to_tensor
from ..modules.layers import LIFLayer, SpyLIFLayer, ALIFLayer


class SpikesEncoder(torch.nn.Module):
    def __init__(
            self,
            n_steps: int,
            n_units: int,
            spikes_layer_type: Type[Union[LIFLayer, SpyLIFLayer, ALIFLayer]],
            dt: float = 1e-3,
            device: Optional[torch.device] = None,
            **kwargs
    ):
        """
        Constructor for the SpikesEncoder class.

        :param n_steps: Number of steps to encode.
        :param n_units: Number of units in the encoder.
        :param spikes_layer_type: Dynamic of the spikes layer.
        :param dt: Time step of the simulation.
        :param device: Device to use for the encoder.
        :param kwargs: Keyword arguments for the spikes layer.

        :keyword kwargs:
            * <forward_weights>: Forward weights of the spikes layer. Default: eye(n_units).
            * <use_recurrent_connection>: Whether to use a recurrent connection. Default: False.
            * <name>: Name of the spikes layer. Default: "encoder".

        """
        super().__init__()
        self.n_steps = n_steps
        self.const_transform = ConstantValuesTransform(n_steps, batch_wise=True)
        self.spikes_layer_type = spikes_layer_type
        self.dt = dt
        kwargs.setdefault('forward_weights', torch.eye(n_units))
        kwargs.setdefault('use_recurrent_connection', False)
        kwargs.setdefault('name', 'encoder')
        kwargs.setdefault('freeze_weights', False)

        assert "dt" not in kwargs, \
            "dt cannot be specified since it must be the given dt"
        assert "device" not in kwargs, \
            "device cannot be specified since it must be the given device"

        self.spikes_layer = self.spikes_layer_type(
            n_units, n_units,
            dt=dt,
            device=device,
            **kwargs,
        )
        self.spikes_layer.build()

    def to(self, device):
        self.spikes_layer.device = device
        return super().to(device)

    def forward(self, x: Any):
        x = to_tensor(x).to(self.spikes_layer.device)
        if x.ndim < 3:
            x = x.unsqueeze(1)
        x = self.const_transform(x)
        x_spikes, hh = [], None
        for t in range(self.n_steps):
            spikes, hh = self.spikes_layer(x[:, t], hh)
            x_spikes.append(spikes)
        # x_spikes = torch.squeeze(torch.stack(x_spikes, dim=1))
        return torch.stack(x_spikes, dim=1)

    def get_and_reset_regularization_loss(self) -> torch.Tensor:
        """
        Get the regularization loss as a sum of all the regularization losses of the layers. Then reset the
        regularization losses.

        :return: the regularization loss.
        """
        regularization_loss = torch.tensor(0.0, dtype=torch.float32, device=self.spikes_layer.device)
        if hasattr(self.spikes_layer, "get_and_reset_regularization_loss") and callable(
                self.spikes_layer.get_and_reset_regularization_loss
        ):
            regularization_loss += self.spikes_layer.get_and_reset_regularization_loss()
        return regularization_loss


class LIFEncoder(SpikesEncoder):
    def __init__(
            self,
            n_steps: int,
            n_units: int,
            dt: float = 1e-3,
            device: Optional[torch.device] = None,
            spikes_layer_kwargs: Optional[dict] = None,
    ):
        if spikes_layer_kwargs is None:
            spikes_layer_kwargs = {}
        super().__init__(
            n_steps, n_units, LIFLayer, dt, device, **spikes_layer_kwargs,
        )


class SpyLIFEncoder(SpikesEncoder):
    def __init__(
            self,
            n_steps: int,
            n_units: int,
            dt: float = 1e-3,
            device: Optional[torch.device] = None,
            spikes_layer_kwargs: Optional[dict] = None,
    ):
        if spikes_layer_kwargs is None:
            spikes_layer_kwargs = {}
        super().__init__(
            n_steps, n_units, SpyLIFLayer, dt, device, **spikes_layer_kwargs,
        )


class ALIFEncoder(SpikesEncoder):
    def __init__(
            self,
            n_steps: int,
            n_units: int,
            dt: float = 1e-3,
            device: Optional[torch.device] = None,
            spikes_layer_kwargs: Optional[dict] = None,
    ):
        if spikes_layer_kwargs is None:
            spikes_layer_kwargs = {}
        super().__init__(
            n_steps, n_units, ALIFLayer, dt, device, **spikes_layer_kwargs,
        )


