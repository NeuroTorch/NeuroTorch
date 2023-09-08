from typing import Optional, Tuple, Union

import torch
from torch import nn

from .base import BaseNeuronsLayer
from ...dimension import SizeTypes
from ...transforms import to_tensor


class Linear(BaseNeuronsLayer):
    def __init__(
            self,
            input_size: Optional[SizeTypes] = None,
            output_size: Optional[SizeTypes] = None,
            name: Optional[str] = None,
            device: Optional[torch.device] = None,
            **kwargs
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            name=name,
            use_recurrent_connection=False,
            device=device,
            **kwargs
        )
        self.bias_weights = None
        self.activation = self._init_activation(self.kwargs["activation"])

    def _set_default_kwargs(self):
        self.kwargs.setdefault("use_bias", True)
        self.kwargs.setdefault("activation", "identity")

    def extra_repr(self):
        return f"{', bias' if self.kwargs['use_bias'] else ''}, activation:{self.activation.__class__.__name__}"

    def build(self) -> 'Linear':
        if self.kwargs["use_bias"]:
            self.bias_weights = nn.Parameter(
                torch.empty((int(self.output_size),), device=self._device),
                requires_grad=self.requires_grad,
            )
        else:
            self.bias_weights = torch.zeros((int(self.output_size),), dtype=torch.float32, device=self._device)
        super().build()
        self.initialize_weights_()
        return self

    def initialize_weights_(self):
        super().initialize_weights_()
        if self.kwargs.get("bias_weights", None) is not None:
            self.bias_weights.data = to_tensor(self.kwargs["bias_weights"]).to(self.device)
        else:
            torch.nn.init.constant_(self.bias_weights, 0.0)

    def create_empty_state(
            self,
            batch_size: int = 1,
            **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        kwargs.setdefault("n_hh", 0)
        return super().create_empty_state(batch_size=batch_size, **kwargs)

    def forward(
            self,
            inputs: torch.Tensor,
            state: Tuple[torch.Tensor, ...] = None,
            **kwargs
    ):
        # assert inputs.ndim == 2
        # batch_size, nb_features = inputs.shape
        return self.activation(torch.matmul(inputs, self.forward_weights) + self.bias_weights)


class LinearRNN(BaseNeuronsLayer):
    def __init__(
            self,
            input_size: Optional[SizeTypes] = None,
            output_size: Optional[SizeTypes] = None,
            name: Optional[str] = None,
            use_recurrent_connection: bool = True,
            device: Optional[torch.device] = None,
            **kwargs
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            name=name,
            use_recurrent_connection=use_recurrent_connection,
            device=device,
            **kwargs
        )
        self.bias_weights = None
        self.activation = self._init_activation(self.kwargs["activation"])

    def _set_default_kwargs(self):
        self.kwargs.setdefault("use_bias", True)
        self.kwargs.setdefault("activation", "identity")

    def extra_repr(self):
        return f"{', bias' if self.kwargs['use_bias'] else ''}, activation:{self.activation.__class__.__name__}"

    def build(self) -> 'LinearRNN':
        if self.kwargs["use_bias"]:
            self.bias_weights = nn.Parameter(
                torch.empty((int(self.output_size),), device=self._device),
                requires_grad=self.requires_grad,
            )
        else:
            self.bias_weights = torch.zeros((int(self.output_size),), dtype=torch.float32, device=self._device)
        super().build()
        self.initialize_weights_()
        return self

    def initialize_weights_(self):
        super().initialize_weights_()
        if self.kwargs.get("bias_weights", None) is not None:
            self.bias_weights.data = to_tensor(self.kwargs["bias_weights"]).to(self.device)
        else:
            torch.nn.init.constant_(self.bias_weights, 0.0)

    def create_empty_state(
            self,
            batch_size: int = 1,
            **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        kwargs.setdefault("n_hh", 1)
        return super().create_empty_state(batch_size=batch_size, **kwargs)

    def forward(
            self,
            inputs: torch.Tensor,
            state: Tuple[torch.Tensor, ...] = None,
            **kwargs
    ):
        # assert inputs.ndim == 2
        batch_size, nb_features = inputs.shape
        V, *_ = self._init_forward_state(state, batch_size, inputs=inputs)
        input_current = torch.matmul(inputs, self.forward_weights)
        if self.use_recurrent_connection:
            rec_current = torch.matmul(V, torch.mul(self.recurrent_weights, self.rec_mask))
        else:
            rec_current = 0.0

        next_V = input_current + rec_current + self.bias_weights
        return self.activation(next_V), (next_V, )
