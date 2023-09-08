from typing import Union, Dict, Any, Tuple, Type, Optional

import torch

from .spikes_decoders import MeanConv
from .spikes_encoders import SpikesEncoder
from ..modules.layers import SpyLIFLayer, LIFLayer, ALIFLayer
from ..modules.base import BaseModel


class SpikesAutoEncoder(BaseModel):

    @staticmethod
    def _default_encoder_type() -> Type[Union[LIFLayer, SpyLIFLayer, ALIFLayer]]:
        return SpyLIFLayer

    def __init__(
            self,
            n_units: int,
            n_encoder_steps: int,
            encoder_type: Optional[Type[Union[LIFLayer, SpyLIFLayer, ALIFLayer]]] = None,
            spikes_encoder: Optional[torch.nn.Module] = None,
            spikes_decoder: Optional[torch.nn.Module] = None,
            **kwargs
    ):
        if encoder_type is not None and spikes_encoder is not None:
            raise ValueError(
                "If encoder_type is provided, spikes_encoder must be None and vice versa."
            )
        kwargs.setdefault("name", "SpikesAutoEncoder")
        super().__init__(n_units, n_units, **kwargs)
        if encoder_type is None:
            encoder_type = self._default_encoder_type()
        self.encoder_type = encoder_type
        self.n_units = n_units
        self.n_encoder_steps = n_encoder_steps
        if spikes_encoder is None:
            spikes_encoder = self._create_default_encoder()
        self.spikes_encoder = spikes_encoder
        if spikes_decoder is None:
            spikes_decoder = self._create_default_decoder()
        self.spikes_decoder = spikes_decoder

    @BaseModel.device.setter
    def device(self, device: torch.device):
        self.spikes_encoder.to(device)
        self.spikes_decoder.to(device)
        BaseModel.device.fset(self, device)

    def _create_default_encoder(self):
        return SpikesEncoder(
            n_steps=self.n_encoder_steps,
            n_units=self.n_units,
            spikes_layer_type=self.encoder_type,
            device=self.device,
            dt=self.kwargs.pop("dt", 1e-3),
            **self.kwargs,
        )

    def _create_default_decoder(self):
        return MeanConv(
            self.n_encoder_steps,
            alpha=2.0,
            learn_alpha=True,
            learn_kernel=True,
            activation=torch.nn.Hardtanh(
                min_val=self.kwargs.get("hard_tanh.min_val", 0.0),
                max_val=self.kwargs.get("hard_tanh.max_val", 1.0)
            ),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.spikes_encoder(x)

    def decode(self, x: torch.Tensor):
        return self.spikes_decoder(x)

    def forward(self, x: torch.Tensor, **kwargs):
        x = self._to_device_transform(x)
        return self.decode(self.encode(x))

    def get_prediction_trace(
            self, inputs: Union[Dict[str, Any], torch.Tensor],
            **kwargs
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        raise NotImplementedError()

    def get_raw_prediction(
            self,
            inputs: torch.Tensor,
            *args,
            **kwargs
    ) -> Union[Tuple[Any, Any, Any], Tuple[Any, Any], Any]:
        return self.forward(inputs)

    def get_and_reset_regularization_loss(self) -> torch.Tensor:
        """
        Get the regularization loss as a sum of all the regularization losses of the layers. Then reset the
        regularization losses.

        :return: the regularization loss.
        """
        regularization_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        for layer in [self.spikes_encoder, self.spikes_decoder]:
            if hasattr(layer, "get_and_reset_regularization_loss") and callable(
                    layer.get_and_reset_regularization_loss
            ):
                regularization_loss += layer.get_and_reset_regularization_loss()
        return regularization_loss








