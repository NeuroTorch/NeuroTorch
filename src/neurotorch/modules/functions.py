import numpy as np
import torch

from ..transforms.base import to_tensor


class WeirdTanh(torch.nn.Module):
    r"""
    Applies the Hyperbolic Tangent (Tanh) function element-wise.

    Tanh is defined as:

    .. math::
        \text{Tanh}(x) = \tanh(x) = \frac{a\exp(\alpha x) - b\exp(-\beta x)} {c\exp(\gamma x) + d\exp(-\delta x)}
    """
    def __init__(
            self,
            a: float = 1.0, b: float = 1.0, c: float = 1.0, d: float = 1.0,
            alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0, delta: float = 1.0,
    ):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def forward(self, x):
        numerator = self.a * torch.exp(self.alpha * x) - self.b * torch.exp(-self.beta * x)
        denominator = self.c * torch.exp(self.gamma * x) + self.d * torch.exp(-self.delta * x)
        return numerator / denominator


class PSigmoid(torch.nn.Module):
    r"""
    Applies the Pseudo-Sigmoid function element-wise.

    Pseudo-Sigmoid is defined as:

    .. math::
        \\text{PSigmoid}(x) = \\frac{1}{1 + \\exp(- p \\odot x)}
    """
    def __init__(self, p: float = 1.0, learn_p: bool = True):
        super().__init__()
        self.p_sqrt = torch.nn.Parameter(to_tensor(np.sqrt(p)), requires_grad=learn_p)

    @property
    def p(self):
        return torch.pow(self.p_sqrt, 2)

    def forward(self, x):
        return 1 / (1 + torch.exp(-self.p * x))
