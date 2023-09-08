from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

from .base import BaseNeuronsLayer
from ...dimension import SizeTypes
from ...transforms import to_tensor


# @inherit_fields_docstring(fields=["Attributes"], bases=[BaseNeuronsLayer])
class LILayer(BaseNeuronsLayer):
    """
    The integration in time of these dynamics is done using the equation
    :eq:`li_v` inspired by Bellec and al. :cite:t:`bellec_solution_2020`.

    .. math::
        :label: li_v

        \\begin{equation}
            V_j^{t+\\Delta t} = \\kappa V_j^{t} + \\sum_{i}^N W_{ij}x_i^{t+\\Delta t} + b_j
        \\end{equation}

    .. math::
        :label: li_kappa

        \\begin{equation}
            \\kappa = e^{-\\frac{\\Delta t}{\\tau_{\\text{mem}}}}
        \\end{equation}


    The parameters of the equation :eq:`li_v` are:

        - :math:`N` is the number of neurons in the layer.
        - :math:`V_j^t` is the synaptic potential of the neuron :math:`j` at time :math:`t`.
        - :math:`\\Delta t` is the integration time step.
        - :math:`\\kappa` is the decay constant of the synaptic current over time (equation :eq:`li_kappa`).
        - :math:`W_{ij}^{\\text{rec}}` is the recurrent weight of the neuron :math:`i` to the neuron :math:`j`.
        - :math:`W_{ij}^{\\text{in}}` is the input weight of the neuron :math:`i` to the neuron :math:`j`.
        - :math:`x_i^{t}` is the input of the neuron :math:`i` at time :math:`t`.

    :Attributes:
        - :attr:`bias_weights` (torch.nn.Parameter): Bias weights of the layer.
        - :attr:`kappa` (torch.nn.Parameter): Decay constant of the synaptic current over time see equation :eq:`li_kappa`.

    """

    def __init__(
            self,
            input_size: Optional[SizeTypes] = None,
            output_size: Optional[SizeTypes] = None,
            name: Optional[str] = None,
            dt: float = 1e-3,
            device: Optional[torch.device] = None,
            **kwargs
    ):
        super(LILayer, self).__init__(
            input_size=input_size,
            output_size=output_size,
            name=name,
            use_recurrent_connection=False,
            dt=dt,
            device=device,
            **kwargs
        )
        self.bias_weights = None
        self.kappa = torch.nn.Parameter(
            torch.tensor(self.kwargs["kappa"], dtype=torch.float32, device=self.device),
            requires_grad=self.kwargs["learn_kappa"]
        )

    def _set_default_kwargs(self):
        self.kwargs.setdefault("tau_out", 10.0 * self.dt)
        self.kwargs.setdefault("kappa", np.exp(-self.dt / self.kwargs["tau_out"]))
        self.kwargs.setdefault("learn_kappa", False)
        self.kwargs.setdefault("use_bias", True)

    def build(self) -> 'LILayer':
        if self.kwargs["use_bias"]:
            self.bias_weights = nn.Parameter(
                torch.empty((int(self.output_size),), device=self._device),
                requires_grad=self.requires_grad,
            )
        else:
            self.bias_weights = torch.zeros((int(self.output_size),), dtype=torch.float32, device=self._device)
        super(LILayer, self).build()
        self.initialize_weights_()
        return self

    def initialize_weights_(self):
        super(LILayer, self).initialize_weights_()
        if self.kwargs.get("bias_weights", None) is not None:
            self.bias_weights.data = to_tensor(self.kwargs["bias_weights"]).to(self.device)
        else:
            torch.nn.init.constant_(self.bias_weights, 0.0)

    def create_empty_state(
            self,
            batch_size: int = 1,
            **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Create an empty state in the following form:
            [membrane potential of shape (batch_size, self.output_size)]

        :param batch_size: The size of the current batch.
        :return: The current state.
        """
        kwargs.setdefault("n_hh", 1)
        return super(LILayer, self).create_empty_state(batch_size=batch_size, **kwargs)

    def forward(
            self,
            inputs: torch.Tensor,
            state: Tuple[torch.Tensor, ...] = None,
            **kwargs
    ):
        assert inputs.ndim == 2
        batch_size, nb_features = inputs.shape
        V, = self._init_forward_state(state, batch_size, inputs=inputs)
        next_V = self.kappa * V + torch.matmul(inputs, self.forward_weights) + self.bias_weights
        return self.activation(next_V), (next_V,)

    def extra_repr(self) -> str:
        _repr = super(LILayer, self).extra_repr()
        _repr += f", bias={self.kwargs['use_bias']}"
        if self.kwargs['learn_kappa']:
            _repr += f", learn_kappa={self.kwargs['learn_kappa']}"
        else:
            _repr += f", kappa={self.kappa.item():.2f}"
        _repr += f", activation={self.activation.__class__.__name__}"
        return _repr


# @inherit_fields_docstring(fields=["Attributes"], bases=[BaseNeuronsLayer])
class SpyLILayer(BaseNeuronsLayer):
    """
    The SpyLI dynamics is a more complex variant of the LI dynamics (class :class:`LILayer`) allowing it to have a
    greater power of expression. This variant is also inspired by Neftci :cite:t:`neftci_surrogate_2019` and also
    contains  two differential equations like the SpyLIF dynamics :class:`SpyLIFLayer`. The equation :eq:`SpyLI_I`
    presents the synaptic current update equation with euler integration while the equation :eq:`SpyLI_V` presents the
    synaptic potential update.

    .. math::
        :label: SpyLI_I

        \\begin{equation}
            I_{\\text{syn}, j}^{t+\\Delta t} = \\alpha I_{\text{syn}, j}^{t} +
            \\sum_{i}^{N} W_{ij}^{\\text{rec}} I_{\\text{syn}, j}^{t}
            + \\sum_i^{N} W_{ij}^{\\text{in}} x_i^{t+\\Delta t}
        \\end{equation}


    .. math::
        :label: SpyLI_V

        \\begin{equation}
            V_j^{t+\\Delta t} = \\beta V_j^t + I_{\\text{syn}, j}^{t+\\Delta t} + b_j
        \\end{equation}


    .. math::
        :label: spyli_alpha

        \\begin{equation}
            \\alpha = e^{-\\frac{\\Delta t}{\\tau_{\\text{syn}}}}
        \\end{equation}

    with :math:`\\tau_{\\text{syn}}` being the decay time constant of the synaptic current.

    .. math::
        :label: spyli_beta

        \\begin{equation}
            \\beta = e^{-\\frac{\\Delta t}{\\tau_{\\text{mem}}}}
        \\end{equation}

    with :math:`\\tau_{\\text{syn}}` being the decay time constant of the synaptic current.

    SpyTorch library: https://github.com/surrogate-gradient-learning/spytorch.

    The variables of the equations :eq:`SpyLI_I` and :eq:`SpyLI_V` are described by the following definitions:

        - :math:`N` is the number of neurons in the layer.
        - :math:`I_{\\text{syn}, j}^{t}` is the synaptic current of neuron :math:`j` at time :math:`t`.
        - :math:`V_j^t` is the synaptic potential of the neuron :math:`j` at time :math:`t`.
        - :math:`\\Delta t` is the integration time step.
        - :math:`\\alpha` is the decay constant of the synaptic current over time (equation :eq:`spyli_alpha`).
        - :math:`\\beta` is the decay constant of the membrane potential over time (equation :eq:`spyli_beta`).
        - :math:`W_{ij}^{\\text{rec}}` is the recurrent weight of the neuron :math:`i` to the neuron :math:`j`.
        - :math:`W_{ij}^{\\text{in}}` is the input weight of the neuron :math:`i` to the neuron :math:`j`.
        - :math:`x_i^{t}` is the input of the neuron :math:`i` at time :math:`t`.

    :Attributes:
        - :attr:`alpha` (torch.nn.Parameter): Decay constant of the synaptic current over time (equation :eq:`spyli_alpha`).
        - :attr:`beta` (torch.nn.Parameter): Decay constant of the membrane potential over time (equation :eq:`spyli_beta`).
        - :attr:`gamma` (torch.nn.Parameter): Slope of the Heaviside function (:math:`\\gamma`).
    """

    def __init__(
            self,
            input_size: Optional[SizeTypes] = None,
            output_size: Optional[SizeTypes] = None,
            name: Optional[str] = None,
            dt: float = 1e-3,
            device: Optional[torch.device] = None,
            **kwargs
    ):
        super(SpyLILayer, self).__init__(
            input_size=input_size,
            output_size=output_size,
            name=name,
            use_recurrent_connection=False,
            dt=dt,
            device=device,
            **kwargs
        )
        self.bias_weights = None
        self.alpha = torch.nn.Parameter(
            torch.tensor(self.kwargs["alpha"], dtype=torch.float32, device=self.device),
            requires_grad=self.kwargs["learn_alpha"]
        )
        self.beta = torch.nn.Parameter(
            torch.tensor(self.kwargs["beta"], dtype=torch.float32, device=self.device),
            requires_grad=self.kwargs["learn_beta"]
        )

    def _set_default_kwargs(self):
        self.kwargs.setdefault("tau_syn", 5.0 * self.dt)
        self.kwargs.setdefault("alpha", np.exp(-self.dt / self.kwargs["tau_syn"]))
        self.kwargs.setdefault("learn_alpha", False)
        self.kwargs.setdefault("tau_mem", 10.0 * self.dt)
        self.kwargs.setdefault("beta", np.exp(-self.dt / self.kwargs["tau_mem"]))
        self.kwargs.setdefault("learn_beta", False)
        self.kwargs.setdefault("use_bias", False)

    def build(self) -> 'SpyLILayer':
        super(SpyLILayer, self).build()
        if self.kwargs["use_bias"]:
            self.bias_weights = torch.nn.Parameter(
                torch.empty((int(self.output_size),), device=self._device),
                requires_grad=self.requires_grad,
            )
        self.initialize_weights_()
        return self

    def initialize_weights_(self):
        super(SpyLILayer, self).initialize_weights_()
        weight_scale = 0.2
        if self.kwargs.get("forward_weights", None) is not None:
            self.forward_weights.data = to_tensor(self.kwargs["forward_weights"]).to(self.device)
        else:
            torch.nn.init.normal_(self.forward_weights, mean=0.0, std=weight_scale / np.sqrt(int(self.input_size)))
        if self.kwargs["use_bias"]:
            if self.kwargs.get("bias_weights", None) is not None:
                self.bias_weights.data = to_tensor(self.kwargs["bias_weights"]).to(self.device)
            else:
                torch.nn.init.constant_(self.bias_weights, 0.0)

    def create_empty_state(
            self,
            batch_size: int = 1,
            **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Create an empty state in the following form:
            [membrane potential of shape (batch_size, self.output_size),
            synaptic current of shape (batch_size, self.output_size)]

        :param batch_size: The size of the current batch.
        :return: The current state.
        """
        kwargs.setdefault("n_hh", 2)
        return super(SpyLILayer, self).create_empty_state(batch_size=batch_size, **kwargs)

    def forward(
            self,
            inputs: torch.Tensor,
            state: Tuple[torch.Tensor, ...] = None,
            **kwargs
    ):
        assert inputs.ndim == 2
        batch_size, nb_features = inputs.shape
        V, I_syn = self._init_forward_state(state, batch_size, inputs=inputs)
        next_I_syn = self.alpha * I_syn + torch.matmul(inputs, self.forward_weights)
        if self.bias_weights is not None:
            next_V = self.beta * V + next_I_syn + self.bias_weights
        else:
            next_V = self.beta * V + next_I_syn
        return self.activation(next_V), (next_V, next_I_syn)

    def extra_repr(self) -> str:
        _repr = super().extra_repr()
        _repr += f", bias={self.kwargs['use_bias']}"
        if self.kwargs['learn_alpha']:
            _repr += f", learn_alpha={self.kwargs['learn_alpha']}"
        else:
            _repr += f", alpha={self.alpha.item():.2f}"
        if self.kwargs['learn_beta']:
            _repr += f", learn_beta={self.kwargs['learn_beta']}"
        else:
            _repr += f", beta={self.beta.item():.2f}"
        _repr += f", activation={self.activation.__class__.__name__}"
        return _repr
