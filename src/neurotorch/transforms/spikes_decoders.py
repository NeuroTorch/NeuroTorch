from typing import Union, Optional

import numpy as np
import torch
import torch.nn.functional

from .base import to_tensor


class MeanConv(torch.nn.Module):
    """
    Apply a weighted mean filter to the input tensor.

    """
    def __init__(
            self,
            kernel_size: int,
            alpha: Union[float, torch.Tensor, np.ndarray] = 1.0,
            learn_alpha: bool = True,
            learn_kernel: bool = True,
            activation: torch.nn.Module = torch.nn.Identity(),
            pad_value: Optional[float] = 0.0,
            pad_mode: str = "left",
    ):
        """
        Constructor for MeanConv class.

        :param kernel_size: Size of the convolution kernel.
        :param alpha: Multiplicative factor for the convolution kernel.
        :param learn_alpha: Whether to learn the alpha parameter.
        :param learn_kernel: Whether to learn the convolution kernel.
        :param activation: Activation function to apply to the output.
        :param pad_value: Value to use for padding the input tensor. This is used to ensure that the input tensor has
            a valid shape for the convolution operation. If None, the input tensor is not padded.
        :param pad_mode: Padding mode. If "left", the padding is applied to the left of the input tensor. If "right",
            the padding is applied to the right of the input tensor. If "both", the padding is applied to both sides
            of the input tensor.
        """
        super(MeanConv, self).__init__()
        self.kernel_size = kernel_size
        self.learn_kernel = learn_kernel
        self.kernel = torch.nn.Parameter(torch.ones(1, self.kernel_size, 1), requires_grad=learn_kernel)
        self.learn_alpha = learn_alpha
        self.alpha = torch.nn.Parameter(to_tensor(alpha), requires_grad=learn_alpha)
        self.activation = activation
        self.pad_value = pad_value
        self.pad_mode = pad_mode

    def extra_repr(self):
        extra_repr = f"kernel(size={self.kernel_size}, learn={self.learn_kernel})"
        if self.learn_alpha:
            extra_repr += f"\nalpha(size={tuple(self.alpha.shape)} learn={self.learn_alpha})"
        else:
            if torch.numel(self.alpha) == 1:
                extra_repr += f"\nalpha(value={self.alpha.detach().cpu().item()})"
            else:
                extra_repr += f"\nalpha(size={tuple(self.alpha.shape)})"
        return extra_repr

    def forward(self, inputs: torch.Tensor):
        batch_size, time_steps, n_units = inputs.shape
        n_pad = (self.kernel_size - (time_steps % self.kernel_size)) % self.kernel_size
        if self.pad_value is not None:
            if self.pad_mode == "left":
                pad = (0, 0, n_pad, 0)
            elif self.pad_mode == "right":
                pad = (0, 0, 0, n_pad)
            elif self.pad_mode == "both":
                pad = (0, 0, int(n_pad - n_pad // 2), int(n_pad // 2))
            else:
                raise ValueError(f"Invalid padding mode: {self.pad_mode}")
            inputs = torch.nn.functional.pad(inputs, pad=pad, value=self.pad_value)

        inputs_view = torch.reshape(inputs, (batch_size, -1, self.kernel_size, n_units))
        inputs_mean = self.alpha * torch.sum(self.kernel * inputs_view, dim=2) / self.kernel_size
        return self.activation(inputs_mean)
	





