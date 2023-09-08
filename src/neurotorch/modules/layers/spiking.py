from typing import Any, Optional, Tuple, Type

import numpy as np
import torch
from torch import nn

from .base import BaseNeuronsLayer
from .. import HeavisideSigmoidApprox, SpikeFunction, HeavisidePhiApprox
from ...dimension import SizeTypes
from ...transforms import to_tensor
from ...utils import format_pseudo_rn_seed


# @inherit_fields_docstring(fields=["Attributes"], bases=[BaseNeuronsLayer])
class LIFLayer(BaseNeuronsLayer):
    """
    LIF dynamics, inspired by :cite:t:`neftci_surrogate_2019` , :cite:t:`bellec_solution_2020` , models the synaptic
    potential and impulses of a neuron over time. The shape of this potential is not considered realistic
    :cite:t:`izhikevich_dynamical_2007` , but the time at which the potential exceeds the threshold is.
    This potential is found by the recurrent equation :eq:`lif_V` .

    .. math::
        \\begin{equation}
            V_j^{t+\\Delta t} = \\left(\\alpha V_j^t + \\sum_{i}^{N} W_{ij}^{\\text{rec}} z_i^t +
            \\sum_i^{N} W_{ij}^{\\text{in}} x_i^{t+\\Delta t}\\right) \\left(1 - z_j^t\\right)
        \\end{equation}
        :label: lif_V


    The variables of the equation :eq:`lif_V` are described by the following definitions:

        - :math:`N` is the number of neurons in the layer.
        - :math:`V_j^t` is the synaptic potential of the neuron :math:`j` at time :math:`t`.
        - :math:`\\Delta t` is the integration time step.
        - :math:`z_j^t` is the spike of the neuron :math:`j` at time :math:`t`.
        - :math:`\\alpha` is the decay constant of the potential over time (equation :eq:`lif_alpha` ).
        - :math:`W_{ij}^{\\text{rec}}` is the recurrent weight of the neuron :math:`i` to the neuron :math:`j`.
        - :math:`W_{ij}^{\\text{in}}` is the input weight of the neuron :math:`i` to the neuron :math:`j`.
        - :math:`x_i^{t}` is the input of the neuron :math:`i` at time :math:`t`.

    .. math::
        :label: lif_alpha

        \\begin{equation}
            \\alpha = e^{-\\frac{\\Delta t}{\\tau_m}}
        \\end{equation}



    with :math:`\\tau_m` being the decay time constant of the membrane potential which is generally 20 ms.

    The output of neuron :math:`j` at time :math:`t` denoted :math:`z_j^t` is defined by the equation :eq:`lif_z` .

    .. math::
        :label: lif_z

        z_j^t = H(V_j^t - V_{\\text{th}})

    where :math:`V_{\\text{th}}` denotes the activation threshold of the neuron and the function :math:`H(\\cdot)`
    is the Heaviside function defined as :math:`H(x) = 1` if :math:`x \\geq 0` and :math:`H(x) = 0` otherwise.

    :Attributes:
        - :attr:`forward_weights` (torch.nn.Parameter): The weights used to compute the output of the layer :math:`W_{ij}^{\\text{in}}` in equation :eq:`lif_V`.
        - :attr:`recurrent_weights` (torch.nn.Parameter): The weights used to compute the hidden state of the layer :math:`W_{ij}^{\\text{rec}}` in equation :eq:`lif_V`.
        - :attr:`dt` (float): The time step of the layer :math:`\\Delta t` in equation :eq:`lif_V`.
        - :attr:`use_rec_eye_mask` (bool): Whether to use the recurrent eye mask.
        - :attr:`rec_mask` (torch.Tensor): The recurrent eye mask.
        - :attr:`alpha` (torch.nn.Parameter): The decay constant of the potential over time. See equation :eq:`lif_alpha` .
        - :attr:`threshold` (torch.nn.Parameter): The activation threshold of the neuron.
        - :attr:`gamma` (torch.nn.Parameter): The gain of the neuron. The gain will increase the gradient of the neuron's output.

    """

    # @inherit_docstring(bases=BaseNeuronsLayer)
    def __init__(
            self,
            input_size: Optional[SizeTypes] = None,
            output_size: Optional[SizeTypes] = None,
            name: Optional[str] = None,
            use_recurrent_connection: bool = True,
            use_rec_eye_mask: bool = False,
            spike_func: Type[SpikeFunction] = HeavisideSigmoidApprox,
            dt: float = 1e-3,
            device: Optional[torch.device] = None,
            **kwargs
    ):
        """
        :keyword float tau_m: The decay time constant of the membrane potential which is generally 20 ms. See equation
            :eq:`lif_alpha` .
        :keyword float threshold: The activation threshold of the neuron.
        :keyword float gamma: The gain of the neuron. The gain will increase the gradient of the neuron's output.
        :keyword float spikes_regularization_factor: The regularization factor of the spikes.
        """
        self.spike_func = spike_func
        super(LIFLayer, self).__init__(
            input_size=input_size,
            output_size=output_size,
            name=name,
            use_recurrent_connection=use_recurrent_connection,
            use_rec_eye_mask=use_rec_eye_mask,
            dt=dt,
            device=device,
            **kwargs
        )

        self.alpha = nn.Parameter(
            torch.tensor(np.exp(-dt / self.kwargs["tau_m"]), dtype=torch.float32, device=self.device),
            requires_grad=False
        )
        self.threshold = nn.Parameter(
            torch.tensor(self.kwargs["threshold"], dtype=torch.float32, device=self.device),
            requires_grad=False
        )
        self.gamma = nn.Parameter(
            torch.tensor(self.kwargs["gamma"], dtype=torch.float32, device=self.device),
            requires_grad=False
        )

    def _set_default_kwargs(self):
        self.kwargs.setdefault("tau_m", 10.0 * self.dt)
        self.kwargs.setdefault("threshold", 1.0)
        if issubclass(self.spike_func, HeavisideSigmoidApprox):
            self.kwargs.setdefault("gamma", 100.0)
        else:
            self.kwargs.setdefault("gamma", 1.0)
        self.kwargs.setdefault("spikes_regularization_factor", 0.0)

    # @inherit_docstring(bases=BaseNeuronsLayer)
    def initialize_weights_(self):
        super().initialize_weights_()
        if self.kwargs.get("forward_weights", None) is not None:
            self.forward_weights.data = to_tensor(self.kwargs["forward_weights"]).to(self.device)
        else:
            torch.nn.init.xavier_normal_(self.forward_weights)

        if self.kwargs.get("recurrent_weights", None) is not None and self.use_recurrent_connection:
            self.recurrent_weights.data = to_tensor(self.kwargs["recurrent_weights"]).to(self.device)
        elif self.use_recurrent_connection:
            torch.nn.init.xavier_normal_(self.recurrent_weights)

    def create_empty_state(
            self,
            batch_size: int = 1,
            **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Create an empty state in the following form:
            ([membrane potential of shape (batch_size, self.output_size)],
            [spikes of shape (batch_size, self.output_size)])

        :param batch_size: The size of the current batch.
        :return: The current state.
        """
        kwargs.setdefault("n_hh", 2)
        return super(LIFLayer, self).create_empty_state(batch_size=batch_size, **kwargs)

    def update_regularization_loss(self, state: Optional[Any] = None, *args, **kwargs) -> torch.Tensor:
        """
        Update the regularization loss for this layer. Each update call increments the regularization loss so at the end
        the regularization loss will be the sum of all calls to this function.

        :param state: The current state of the layer.
        :return: The updated regularization loss.
        """
        next_V, next_Z = state
        self._regularization_loss += self.kwargs["spikes_regularization_factor"] * torch.sum(next_Z)
        # self._regularization_loss += 2e-6*torch.mean(torch.sum(next_Z, dim=-1)**2)
        return self._regularization_loss

    # @inherit_docstring(bases=BaseNeuronsLayer)
    def forward(
            self,
            inputs: torch.Tensor,
            state: Tuple[torch.Tensor, ...] = None,
            **kwargs
    ):
        assert inputs.ndim == 2
        batch_size, nb_features = inputs.shape
        V, Z = self._init_forward_state(state, batch_size, inputs=inputs)
        input_current = torch.matmul(inputs, self.forward_weights)
        if self.use_recurrent_connection:
            rec_current = torch.matmul(Z, torch.mul(self.recurrent_weights, self.rec_mask))
        else:
            rec_current = 0.0
        next_V = (self.alpha * V + input_current + rec_current) * (1.0 - Z.detach())
        next_Z = self.spike_func.apply(next_V, self.threshold, self.gamma)
        return next_Z, (next_V, next_Z)


# @inherit_fields_docstring(fields=["Attributes"], bases=[BaseNeuronsLayer])
class SpyLIFLayer(BaseNeuronsLayer):
    """
    The SpyLIF dynamics is a more complex variant of the LIF dynamics (class :class:`LIFLayer`) allowing it to have a
    greater power of expression. This variant is also inspired by Neftci :cite:t:`neftci_surrogate_2019` and also
    contains  two differential equations like the SpyLI dynamics :class:`SpyLI`. The equation :eq:`SpyLIF_I` presents
    the synaptic current update equation with euler integration while the equation :eq:`SpyLIF_V` presents the
    synaptic potential update.

    .. math::
        :label: SpyLIF_I

        \\begin{equation}
            I_{\\text{syn}, j}^{t+\\Delta t} = \\alpha I_{\text{syn}, j}^{t} + \\sum_{i}^{N} W_{ij}^{\\text{rec}} z_i^t
            + \\sum_i^{N} W_{ij}^{\\text{in}} x_i^{t+\\Delta t}
        \\end{equation}


    .. math::
        :label: SpyLIF_V

        \\begin{equation}
            V_j^{t+\\Delta t} = \\left(\\beta V_j^t + I_{\\text{syn}, j}^{t+\\Delta t}\\right) \\left(1 - z_j^t\\right)
        \\end{equation}


    .. math::
        :label: spylif_alpha

        \\begin{equation}
            \\alpha = e^{-\\frac{\\Delta t}{\\tau_{\\text{syn}}}}
        \\end{equation}

    with :math:`\\tau_{\\text{syn}}` being the decay time constant of the synaptic current.

    .. math::
        :label: spylif_beta

        \\begin{equation}
            \\beta = e^{-\\frac{\\Delta t}{\\tau_{\\text{mem}}}}
        \\end{equation}

    with :math:`\\tau_{\\text{syn}}` being the decay time constant of the synaptic current.

    The output of neuron :math:`j` at time :math:`t` denoted :math:`z_j^t` is defined by the equation :eq:`spylif_z` .

    .. math::
        :label: spylif_z

        z_j^t = H(V_j^t - V_{\\text{th}})

    where :math:`V_{\\text{th}}` denotes the activation threshold of the neuron and the function :math:`H(\\cdot)`
    is the Heaviside function defined as :math:`H(x) = 1` if :math:`x \\geq 0` and :math:`H(x) = 0` otherwise.

    SpyTorch library: https://github.com/surrogate-gradient-learning/spytorch.

    The variables of the equations :eq:`SpyLIF_I` and :eq:`SpyLIF_V` are described by the following definitions:

        - :math:`N` is the number of neurons in the layer.
        - :math:`I_{\\text{syn}, j}^{t}` is the synaptic current of neuron :math:`j` at time :math:`t`.
        - :math:`V_j^t` is the synaptic potential of the neuron :math:`j` at time :math:`t`.
        - :math:`\\Delta t` is the integration time step.
        - :math:`z_j^t` is the spike of the neuron :math:`j` at time :math:`t`.
        - :math:`\\alpha` is the decay constant of the synaptic current over time (equation :eq:`spylif_alpha`).
        - :math:`\\beta` is the decay constant of the membrane potential over time (equation :eq:`spylif_beta`).
        - :math:`W_{ij}^{\\text{rec}}` is the recurrent weight of the neuron :math:`i` to the neuron :math:`j`.
        - :math:`W_{ij}^{\\text{in}}` is the input weight of the neuron :math:`i` to the neuron :math:`j`.
        - :math:`x_i^{t}` is the input of the neuron :math:`i` at time :math:`t`.

    :Attributes:
        - :attr:`alpha` (torch.nn.Parameter): Decay constant of the synaptic current over time (equation :eq:`spylif_alpha`).
        - :attr:`beta` (torch.nn.Parameter): Decay constant of the membrane potential over time (equation :eq:`spylif_beta`).
        - :attr:`threshold` (torch.nn.Parameter): Activation threshold of the neuron (:math:`V_{\\text{th}}`).
        - :attr:`gamma` (torch.nn.Parameter): Slope of the Heaviside function (:math:`\\gamma`).
    """

    def __init__(
            self,
            input_size: Optional[SizeTypes] = None,
            output_size: Optional[SizeTypes] = None,
            name: Optional[str] = None,
            use_recurrent_connection: bool = True,
            use_rec_eye_mask: bool = False,
            dt: float = 1e-3,
            device: Optional[torch.device] = None,
            **kwargs
    ):
        """
        Constructor for the SpyLIF layer.

        :param input_size: The size of the input.
        :type input_size: Optional[SizeTypes]
        :param output_size: The size of the output.
        :type output_size: Optional[SizeTypes]
        :param name: The name of the layer.
        :type name: Optional[str]
        :param use_recurrent_connection: Whether to use the recurrent connection.
        :type use_recurrent_connection: bool
        :param use_rec_eye_mask: Whether to use the recurrent eye mask.
        :type use_rec_eye_mask: bool
        :param spike_func: The spike function to use.
        :type spike_func: Callable[[torch.Tensor], torch.Tensor]
        :param learning_type: The learning type to use.
        :type learning_type: LearningType
        :param dt: Time step (Euler's discretisation).
        :type dt: float
        :param device: The device to use.
        :type device: Optional[torch.device]
        :param kwargs: The keyword arguments for the layer.

        :keyword float tau_syn: The synaptic time constant :math:`\\tau_{\\text{syn}}`. Default: 5.0 * dt.
        :keyword float tau_mem: The membrane time constant :math:`\\tau_{\\text{mem}}`. Default: 10.0 * dt.
        :keyword float threshold: The threshold potential :math:`V_{\\text{th}}`. Default: 1.0.
        :keyword float gamma: The multiplier of the derivative of the spike function :math:`\\gamma`. Default: 100.0.
        :keyword float spikes_regularization_factor: The regularization factor for the spikes. Higher this factor is,
            the more the network will tend to spike less. Default: 0.0.

        """
        self.spike_func = HeavisideSigmoidApprox
        super(SpyLIFLayer, self).__init__(
            input_size=input_size,
            output_size=output_size,
            name=name,
            use_recurrent_connection=use_recurrent_connection,
            use_rec_eye_mask=use_rec_eye_mask,
            dt=dt,
            device=device,
            **kwargs
        )

        self.alpha = nn.Parameter(
            torch.tensor(np.exp(-dt / self.kwargs["tau_syn"]), dtype=torch.float32, device=self.device),
            requires_grad=False
        )
        self.beta = nn.Parameter(
            torch.tensor(np.exp(-dt / self.kwargs["tau_mem"]), dtype=torch.float32, device=self.device),
            requires_grad=False
        )
        self.threshold = nn.Parameter(
            torch.tensor(self.kwargs["threshold"], dtype=torch.float32, device=self.device),
            requires_grad=False
        )
        self.gamma = nn.Parameter(
            torch.tensor(self.kwargs["gamma"], dtype=torch.float32, device=self.device),
            requires_grad=False
        )
        self._regularization_l1 = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._n_spike_per_neuron = torch.zeros(int(self.output_size), dtype=torch.float32, device=self.device)
        self._total_count = 0

    def _set_default_kwargs(self):
        self.kwargs.setdefault("tau_syn", 5.0 * self.dt)
        self.kwargs.setdefault("tau_mem", 10.0 * self.dt)
        self.kwargs.setdefault("threshold", 1.0)
        self.kwargs.setdefault("gamma", 100.0)
        self.kwargs.setdefault("spikes_regularization_factor", 0.0)
        self.kwargs.setdefault("hh_init", "zeros")

    def initialize_weights_(self):
        super().initialize_weights_()
        weight_scale = 0.2
        if self.kwargs.get("forward_weights", None) is not None:
            self.forward_weights.data = to_tensor(self.kwargs["forward_weights"]).to(self.device)
        else:
            torch.nn.init.normal_(self.forward_weights, mean=0.0, std=weight_scale / np.sqrt(int(self.input_size)))

        if self.kwargs.get("recurrent_weights", None) is not None and self.use_recurrent_connection:
            self.recurrent_weights.data = to_tensor(self.kwargs["recurrent_weights"]).to(self.device)
        elif self.use_recurrent_connection:
            torch.nn.init.normal_(self.recurrent_weights, mean=0.0, std=weight_scale / np.sqrt(int(self.output_size)))

    def create_empty_state(
            self,
            batch_size: int = 1,
            **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Create an empty state in the following form:
            ([membrane potential of shape (batch_size, self.output_size)],
            [synaptic current of shape (batch_size, self.output_size)],
            [spikes of shape (batch_size, self.output_size)])

        :param batch_size: The size of the current batch.
        :return: The current state.
        """
        kwargs.setdefault("n_hh", 3)
        thr = self.threshold.detach().cpu().item()
        if self.kwargs["hh_init"] == "random":
            V_mu, V_std = self.kwargs.get("hh_init_mu", thr / 2.0), self.kwargs.get("hh_init_std", 0.341 * thr)
            gen = torch.Generator(device=self.device)
            gen.manual_seed(format_pseudo_rn_seed(self.kwargs.get("hh_init_seed", None)))
            V = torch.clamp_min(
                torch.rand(
                    (batch_size, int(self.output_size)),
                    device=self.device,
                    dtype=torch.float32,
                    requires_grad=True,
                    generator=gen,
                ) * V_std + V_mu, min=0.0
            )
            I = torch.rand(
                (batch_size, int(self.output_size)),
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
                generator=gen,
            )
            Z = self.spike_func.apply(V, self.threshold, self.gamma)
            V = V * (1.0 - Z)
            return tuple([V, I, Z])
        elif self.kwargs["hh_init"] == "inputs":
            assert "inputs" in kwargs, "The inputs must be provided to initialize the state."
            assert int(self.input_size) == int(self.output_size), \
                "The input and output size must be the same with inputs initialization."
            # V_mu, V_std = self.kwargs.get("hh_init_mu", thr / 2.0), self.kwargs.get("hh_init_std", 0.341 * thr)
            V_mu, V_std = self.kwargs.get("hh_init_mu", 0.0), self.kwargs.get("hh_init_std", thr)
            gen = torch.Generator(device=self.device)
            gen.manual_seed(format_pseudo_rn_seed(self.kwargs.get("hh_init_seed", None)))
            I = torch.rand(
                (batch_size, int(self.output_size)),
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
                generator=gen,
            )
            Z = kwargs["inputs"].clone()
            V = (torch.rand(
                (batch_size, int(self.output_size)),
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
                generator=gen,
            ) * V_std + V_mu)
            V = (self.beta * V + self.alpha * I) * (1.0 - Z)

            return tuple([V, I, Z])
        return super(SpyLIFLayer, self).create_empty_state(batch_size=batch_size, **kwargs)

    def reset_regularization_loss(self):
        super(SpyLIFLayer, self).reset_regularization_loss()
        self._regularization_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._regularization_l1 = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._n_spike_per_neuron = torch.zeros(int(self.output_size), dtype=torch.float32, device=self.device)
        self._total_count = 0

    def update_regularization_loss(self, state: Optional[Any] = None, *args, **kwargs) -> torch.Tensor:
        """
        Update the regularization loss for this layer. Each update call increments the regularization loss so at the end
        the regularization loss will be the sum of all calls to this function.

        :param state: The current state of the layer.
        :return: The updated regularization loss.
        """
        next_V, next_I_syn, next_Z = state
        self._regularization_l1 += self.kwargs["spikes_regularization_factor"] * torch.sum(next_Z)
        # self._n_spike_per_neuron += torch.sum(torch.sum(next_Z, dim=0), dim=0)
        # self._total_count += next_Z.shape[0]*next_Z.shape[1]
        # current_l2 = self.kwargs["spikes_regularization_factor"]*torch.sum(self._n_spike_per_neuron ** 2) / (self._total_count + 1e-6)
        # self._regularization_loss = self._regularization_l1 + current_l2
        self._regularization_loss = self._regularization_l1
        return self._regularization_loss

    def forward(
            self,
            inputs: torch.Tensor,
            state: Tuple[torch.Tensor, ...] = None,
            **kwargs
    ):
        assert inputs.ndim == 2, f"Inputs must be of shape (batch_size, input_size), got {inputs.shape}."
        batch_size, nb_features = inputs.shape
        V, I_syn, Z = self._init_forward_state(state, batch_size, inputs=inputs)
        input_current = torch.matmul(inputs, self.forward_weights)
        if self.use_recurrent_connection:
            rec_current = torch.matmul(Z, torch.mul(self.recurrent_weights, self.rec_mask))
        else:
            rec_current = 0.0
        next_I_syn = self.alpha * I_syn + input_current + rec_current
        next_V = (self.beta * V + next_I_syn) * (1.0 - Z.detach())
        next_Z = self.spike_func.apply(next_V, self.threshold, self.gamma)
        return next_Z, (next_V, next_I_syn, next_Z)


class SpyALIFLayer(SpyLIFLayer):
    """
    The SpyALIF dynamic, inspired by Bellec and \\textit{al.} :cite:t:`bellec_solution_2020` and bye the
    :class:`SpyLIFLayer` from the work of Neftci :cite:t:`neftci_surrogate_2019`, is very
    similar to the SpyLIF dynamics (class :class:`SpyLIFLayer`). In fact, SpyALIF has exactly the same potential
    update equation as SpyLIF. The difference comes
    from the fact that the threshold potential varies with time and neuron input. Indeed, the threshold
    is increased at each output spike and is then decreased with a certain rate in order to come back to
    its starting threshold :math:`V_{\\text{th}}`. The threshold equation from :class:`SpyLIFLayer` is thus slightly
    modified by changing :math:`V_{\\text{th}} \\to A_j^t`. Thus, the output of neuron :math:`j` at time :math:`t`
    denoted :math:`z_j^t` is redefined by the equation :eq:`alif_z`.

    .. math::
        :label: SpyALIF_I

        \\begin{equation}
            I_{\\text{syn}, j}^{t+\\Delta t} = \\alpha I_{\text{syn}, j}^{t} + \\sum_{i}^{N} W_{ij}^{\\text{rec}} z_i^t
            + \\sum_i^{N} W_{ij}^{\\text{in}} x_i^{t+\\Delta t}
        \\end{equation}


    .. math::
        :label: SpyALIF_V

        \\begin{equation}
            V_j^{t+\\Delta t} = \\left(\\beta V_j^t + I_{\\text{syn}, j}^{t+\\Delta t}\\right) \\left(1 - z_j^t\\right)
        \\end{equation}


    .. math::
        :label: spyalif_alpha

        \\begin{equation}
            \\alpha = e^{-\\frac{\\Delta t}{\\tau_{\\text{syn}}}}
        \\end{equation}

    with :math:`\\tau_{\\text{syn}}` being the decay time constant of the synaptic current.

    .. math::
        :label: spyalif_beta

        \\begin{equation}
            \\beta = e^{-\\frac{\\Delta t}{\\tau_{\\text{mem}}}}
        \\end{equation}

    with :math:`\\tau_{\\text{syn}}` being the decay time constant of the synaptic current.

    The output of neuron :math:`j` at time :math:`t` denoted :math:`z_j^t` is defined by the equation :eq:`spyalif_z` .

    .. math::
        :label: spyalif_z

        z_j^t = H(V_j^t - A_j^t)

    where :math:`A_j^t` denotes the activation threshold of the neuron and the function :math:`H(\\cdot)`
    is the Heaviside function defined as :math:`H(x) = 1` if :math:`x \\geq 0` and :math:`H(x) = 0` otherwise.
    The update of the activation threshold is then described by :eq:`alif_A`.

    .. math::
        :label: alif_A

        \\begin{equation}
            A_j^t = V_{\\text{th}} + \\kappa a_j^t
        \\end{equation}

    with the adaptation variable :math:`a_j^t` described by :eq:`alif_a` and :math:`\\kappa` an amplification
    factor greater than 1 and typically equivalent to :math:`\\kappa\\approx 1.6` :cite:t:`bellec_solution_2020`.

    .. math::
        :label: alif_a

        \\begin{equation}
            a_j^{t+1} = \\rho a_j + z_j^t
        \\end{equation}

    With the decay factor :math:`\\rho` as:

    .. math::
        :label: alif_rho

        \\begin{equation}
            \\rho = e^{-\\frac{\\Delta t}{\\tau_a}}
        \\end{equation}

    SpyTorch library: https://github.com/surrogate-gradient-learning/spytorch.

    The variables of the equations :eq:`SpyALIF_I` and :eq:`SpyALIF_V` are described by the following definitions:

        - :math:`N` is the number of neurons in the layer.
        - :math:`I_{\\text{syn}, j}^{t}` is the synaptic current of neuron :math:`j` at time :math:`t`.
        - :math:`V_j^t` is the synaptic potential of the neuron :math:`j` at time :math:`t`.
        - :math:`\\Delta t` is the integration time step.
        - :math:`z_j^t` is the spike of the neuron :math:`j` at time :math:`t`.
        - :math:`\\alpha` is the decay constant of the synaptic current over time (equation :eq:`spylif_alpha`).
        - :math:`\\beta` is the decay constant of the membrane potential over time (equation :eq:`spylif_beta`).
        - :math:`W_{ij}^{\\text{rec}}` is the recurrent weight of the neuron :math:`i` to the neuron :math:`j`.
        - :math:`W_{ij}^{\\text{in}}` is the input weight of the neuron :math:`i` to the neuron :math:`j`.
        - :math:`x_i^{t}` is the input of the neuron :math:`i` at time :math:`t`.

    :Attributes:
        - :attr:`alpha` (torch.nn.Parameter): Decay constant of the synaptic current over time (equation :eq:`spyalif_alpha`).
        - :attr:`beta` (torch.nn.Parameter): Decay constant of the membrane potential over time (equation :eq:`spyalif_beta`).
        - :attr:`threshold` (torch.nn.Parameter): Activation threshold of the neuron (:math:`V_{\\text{th}}`).
        - :attr:`gamma` (torch.nn.Parameter): Slope of the Heaviside function (:math:`\\gamma`).
        - :attr:`kappa`: The amplification factor of the threshold potential (:math:`\\kappa`).
        - :attr:`rho`: The decay factor of the adaptation variable (:math:`\\rho`).
    """

    def __init__(
            self,
            input_size: Optional[SizeTypes] = None,
            output_size: Optional[SizeTypes] = None,
            name: Optional[str] = None,
            use_recurrent_connection: bool = True,
            use_rec_eye_mask: bool = False,
            dt: float = 1e-3,
            device: Optional[torch.device] = None,
            **kwargs
    ):
        """
        Constructor for the SpyLIF layer.

        :param input_size: The size of the input.
        :type input_size: Optional[SizeTypes]
        :param output_size: The size of the output.
        :type output_size: Optional[SizeTypes]
        :param name: The name of the layer.
        :type name: Optional[str]
        :param use_recurrent_connection: Whether to use the recurrent connection.
        :type use_recurrent_connection: bool
        :param use_rec_eye_mask: Whether to use the recurrent eye mask.
        :type use_rec_eye_mask: bool
        :param spike_func: The spike function to use.
        :type spike_func: Callable[[torch.Tensor], torch.Tensor]
        :param learning_type: The learning type to use.
        :type learning_type: LearningType
        :param dt: Time step (Euler's discretisation).
        :type dt: float
        :param device: The device to use.
        :type device: Optional[torch.device]
        :param kwargs: The keyword arguments for the layer.

        :keyword float tau_syn: The synaptic time constant :math:`\\tau_{\\text{syn}}`. Default: 5.0 * dt.
        :keyword float tau_mem: The membrane time constant :math:`\\tau_{\\text{mem}}`. Default: 10.0 * dt.
        :keyword float threshold: The threshold potential :math:`V_{\\text{th}}`. Default: 1.0.
        :keyword float gamma: The multiplier of the derivative of the spike function :math:`\\gamma`. Default: 100.0.
        :keyword float spikes_regularization_factor: The regularization factor for the spikes. Higher this factor is,
            the more the network will tend to spike less. Default: 0.0.

        """
        self.spike_func = HeavisideSigmoidApprox
        super(SpyALIFLayer, self).__init__(
            input_size=input_size,
            output_size=output_size,
            name=name,
            use_recurrent_connection=use_recurrent_connection,
            use_rec_eye_mask=use_rec_eye_mask,
            dt=dt,
            device=device,
            **kwargs
        )

        self.kappa = nn.Parameter(
            torch.tensor(self.kwargs["kappa"], dtype=torch.float32, device=self.device),
            requires_grad=self.kwargs["learn_kappa"]
        )
        self.rho = nn.Parameter(
            torch.tensor(np.exp(-dt / self.kwargs["tau_a"]), dtype=torch.float32, device=self.device),
            requires_grad=False
        )

    def _set_default_kwargs(self):
        self.kwargs.setdefault("tau_syn", 5.0 * self.dt)
        self.kwargs.setdefault("tau_mem", 10.0 * self.dt)
        self.kwargs.setdefault("tau_a", 200.0 * self.dt)
        self.kwargs.setdefault("threshold", 1.0)
        self.kwargs.setdefault("gamma", 100.0)
        self.kwargs.setdefault("kappa", 1.6)
        self.kwargs.setdefault("learn_kappa", False)
        self.kwargs.setdefault("spikes_regularization_factor", 0.0)
        self.kwargs.setdefault("hh_init", "zeros")

    def initialize_weights_(self):
        super().initialize_weights_()
        weight_scale = 0.2
        if self.kwargs.get("forward_weights", None) is not None:
            self.forward_weights.data = to_tensor(self.kwargs["forward_weights"]).to(self.device)
        else:
            torch.nn.init.normal_(self.forward_weights, mean=0.0, std=weight_scale / np.sqrt(int(self.input_size)))

        if self.kwargs.get("recurrent_weights", None) is not None and self.use_recurrent_connection:
            self.recurrent_weights.data = to_tensor(self.kwargs["recurrent_weights"]).to(self.device)
        elif self.use_recurrent_connection:
            torch.nn.init.normal_(self.recurrent_weights, mean=0.0, std=weight_scale / np.sqrt(int(self.output_size)))

    def create_empty_state(
            self,
            batch_size: int = 1,
            **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Create an empty state in the following form:
            ([membrane potential of shape (batch_size, self.output_size)],
            [synaptic current of shape (batch_size, self.output_size)],
            [spikes of shape (batch_size, self.output_size)])

        :param batch_size: The size of the current batch.
        :return: The current state.
        """
        kwargs.setdefault("n_hh", 4)
        thr = self.threshold.detach().cpu().item()
        if self.kwargs["hh_init"] == "random":
            V_mu, V_std = self.kwargs.get("hh_init_mu", thr / 2.0), self.kwargs.get("hh_init_std", 0.341 * thr)
            gen = torch.Generator(device=self.device)
            gen.manual_seed(format_pseudo_rn_seed(self.kwargs.get("hh_init_seed", None)))
            V = torch.clamp_min(
                torch.rand(
                    (batch_size, int(self.output_size)),
                    device=self.device,
                    dtype=torch.float32,
                    requires_grad=True,
                    generator=gen,
                ) * V_std + V_mu, min=0.0
            )
            I = torch.rand(
                (batch_size, int(self.output_size)),
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
                generator=gen,
            )
            Z = self.spike_func.apply(V, self.threshold, self.gamma)
            V = V * (1.0 - Z)
            a = torch.rand(
                (batch_size, int(self.output_size)),
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
                generator=gen,
            )
            return tuple([V, I, a, Z])
        elif self.kwargs["hh_init"] == "inputs":
            # V_mu, V_std = self.kwargs.get("hh_init_mu", thr / 2.0), self.kwargs.get("hh_init_std", 0.341 * thr)
            V_mu, V_std = self.kwargs.get("hh_init_mu", 0.0), self.kwargs.get("hh_init_std", thr)
            gen = torch.Generator(device=self.device)
            gen.manual_seed(format_pseudo_rn_seed(self.kwargs.get("hh_init_seed", None)))
            I = torch.rand(
                (batch_size, int(self.output_size)),
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
                generator=gen,
            )
            Z = kwargs["inputs"].clone()
            V = (torch.rand(
                (batch_size, int(self.output_size)),
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
                generator=gen,
            ) * V_std + V_mu)
            V = (self.beta * V + self.alpha * I) * (1.0 - Z)
            a = self.rho * torch.rand(
                (batch_size, int(self.output_size)),
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
                generator=gen,
            ) * thr + Z
            return tuple([V, I, a, Z])
        return super(SpyLIFLayer, self).create_empty_state(batch_size=batch_size, **kwargs)

    def reset_regularization_loss(self):
        super(SpyLIFLayer, self).reset_regularization_loss()
        self._regularization_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._regularization_l1 = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._n_spike_per_neuron = torch.zeros(int(self.output_size), dtype=torch.float32, device=self.device)
        self._total_count = 0

    def update_regularization_loss(self, state: Optional[Any] = None, *args, **kwargs) -> torch.Tensor:
        """
        Update the regularization loss for this layer. Each update call increments the regularization loss so at the end
        the regularization loss will be the sum of all calls to this function.

        :param state: The current state of the layer.
        :return: The updated regularization loss.
        """
        next_V, next_I_syn, next_Z = state
        self._regularization_l1 += self.kwargs["spikes_regularization_factor"] * torch.sum(next_Z)
        # self._n_spike_per_neuron += torch.sum(torch.sum(next_Z, dim=0), dim=0)
        # self._total_count += next_Z.shape[0]*next_Z.shape[1]
        # current_l2 = self.kwargs["spikes_regularization_factor"]*torch.sum(self._n_spike_per_neuron ** 2) / (self._total_count + 1e-6)
        # self._regularization_loss = self._regularization_l1 + current_l2
        self._regularization_loss = self._regularization_l1
        return self._regularization_loss

    def forward(
            self,
            inputs: torch.Tensor,
            state: Tuple[torch.Tensor, ...] = None,
            **kwargs
    ):
        assert inputs.ndim == 2, f"Inputs must be of shape (batch_size, input_size), got {inputs.shape}."
        batch_size, nb_features = inputs.shape
        V, I_syn, a, Z = self._init_forward_state(state, batch_size, inputs=inputs)
        input_current = torch.matmul(inputs, self.forward_weights)
        if self.use_recurrent_connection:
            rec_current = torch.matmul(Z, torch.mul(self.recurrent_weights, self.rec_mask))
        else:
            rec_current = 0.0
        next_I_syn = self.alpha * I_syn + input_current + rec_current
        next_V = (self.beta * V + next_I_syn) * (1.0 - Z.detach())
        next_a = self.rho * a + Z  # a^{t+1} = \rho * a_j^t + z_j^t
        A = self.threshold + self.kappa * next_a  # A_j^t = v_{th} + \kappa * a_j^t
        next_Z = self.spike_func.apply(next_V, A, self.gamma)  # z_j^t = H(v_j^t - A_j^t)
        return next_Z, (next_V, next_I_syn, next_a, next_Z)


# @inherit_fields_docstring(fields=["Attributes"], bases=[LIFLayer])
class ALIFLayer(LIFLayer):
    """
    The ALIF dynamic, inspired by Bellec and \\textit{al.} :cite:t:`bellec_solution_2020`, is very
    similar to the LIF dynamics (class :class:`LIFLayer`). In fact, ALIF has exactly the same potential
    update equation as LIF. The difference comes
    from the fact that the threshold potential varies with time and neuron input. Indeed, the threshold
    is increased at each output pulse and is then decreased with a certain rate in order to come back to
    its starting threshold :math:`V_{\\text{th}}`. The threshold equation from :class:`LIFLayer` is thus slightly
    modified by changing :math:`V_{\\text{th}} \\to A_j^t`. Thus, the output of neuron :math:`j` at time :math:`t`
    denoted :math:`z_j^t` is redefined by the equation :eq:`alif_z`.

    .. math::
        :label: alif_z

        \\begin{equation}
            z_j^t = H(V_j^t - A_j^t)
        \\end{equation}

    The update of the activation threshold is then described by :eq:`alif_A`.

    .. math::
        :label: alif_A

        \\begin{equation}
            A_j^t = V_{\\text{th}} + \\beta a_j^t
        \\end{equation}

    with the adaptation variable :math:`a_j^t` described by :eq:`alif_a` and :math:`\\beta` an amplification
    factor greater than 1 and typically equivalent to :math:`\\beta\\approx 1.6` :cite:t:`bellec_solution_2020`.

    .. math::
        :label: alif_a

        \\begin{equation}
            a_j^{t+1} = \\rho a_j + z_j^t
        \\end{equation}

    With the decay factor :math:`\\rho` as:

    .. math::
        :label: alif_rho

        \\begin{equation}
            \\rho = e^{-\\frac{\\Delta t}{\\tau_a}}
        \\end{equation}

    :Attributes:
        - :attr:`beta`: The amplification factor of the threshold potential :math:`\\beta`.
        - :attr:`rho`: The decay factor of the adaptation variable :math:`\\rho`.

    """

    def __init__(
            self,
            input_size: Optional[SizeTypes] = None,
            output_size: Optional[SizeTypes] = None,
            name: Optional[str] = None,
            use_recurrent_connection: bool = True,
            use_rec_eye_mask: bool = False,
            spike_func: Type[SpikeFunction] = HeavisideSigmoidApprox,
            dt: float = 1e-3,
            device: Optional[torch.device] = None,
            **kwargs
    ):
        super(ALIFLayer, self).__init__(
            input_size=input_size,
            output_size=output_size,
            name=name,
            use_recurrent_connection=use_recurrent_connection,
            use_rec_eye_mask=use_rec_eye_mask,
            spike_func=spike_func,
            dt=dt,
            device=device,
            **kwargs
        )
        self.beta = nn.Parameter(
            torch.tensor(self.kwargs["beta"], dtype=torch.float32, device=self.device),
            requires_grad=self.kwargs["learn_beta"]
        )
        self.rho = nn.Parameter(
            torch.tensor(np.exp(-dt / self.kwargs["tau_a"]), dtype=torch.float32, device=self.device),
            requires_grad=False
        )

    def _set_default_kwargs(self):
        self.kwargs.setdefault("tau_m", 20.0 * self.dt)
        self.kwargs.setdefault("tau_a", 200.0 * self.dt)
        self.kwargs.setdefault("beta", 1.6)
        # self.kwargs.setdefault("threshold", 0.03)
        self.kwargs.setdefault("threshold", 1.0)
        if issubclass(self.spike_func, HeavisideSigmoidApprox):
            self.kwargs.setdefault("gamma", 100.0)
        else:
            self.kwargs.setdefault("gamma", 0.3)
        self.kwargs.setdefault("learn_beta", False)
        self.kwargs.setdefault("spikes_regularization_factor", 0.0)

    def create_empty_state(
            self,
            batch_size: int = 1,
            **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Create an empty state in the following form:
            [[membrane potential of shape (batch_size, self.output_size)]
            [current threshold of shape (batch_size, self.output_size)]
            [spikes of shape (batch_size, self.output_size)]]

        :param batch_size: The size of the current batch.
        :type batch_size: int

        :return: The current state.
        :rtype: Tuple[torch.Tensor, ...]
        """
        kwargs.setdefault("n_hh", 3)
        return super(ALIFLayer, self).create_empty_state(batch_size=batch_size, **kwargs)

    def forward(
            self,
            inputs: torch.Tensor,
            state: Tuple[torch.Tensor, ...] = None,
            **kwargs
    ):
        assert inputs.ndim == 2
        batch_size, nb_features = inputs.shape
        V, a, Z = self._init_forward_state(state, batch_size, inputs=inputs)
        input_current = torch.matmul(inputs, self.forward_weights)
        if self.use_recurrent_connection:
            rec_current = torch.matmul(Z, torch.mul(self.recurrent_weights, self.rec_mask))
        else:
            rec_current = 0.0
        # v_j^{t+1} = \alpha * v_j^t + \sum_i W_{ji}*z_i^t + \sum_i W_{ji}^{in}x_i^{t+1} - z_j^t * v_{th}
        next_V = (self.alpha * V + input_current + rec_current) * (1.0 - Z.detach())
        next_a = self.rho * a + Z  # a^{t+1} = \rho * a_j^t + z_j^t
        A = self.threshold + self.beta * next_a  # A_j^t = v_{th} + \beta * a_j^t
        next_Z = self.spike_func.apply(next_V, A, self.gamma)  # z_j^t = H(v_j^t - A_j^t)
        return next_Z, (next_V, next_a, next_Z)

    def update_regularization_loss(self, state: Optional[Any] = None, *args, **kwargs) -> torch.Tensor:
        """
        Update the regularization loss for this layer. Each update call increments the regularization loss so at the end
        the regularization loss will be the sum of all calls to this function.

        :param state: The current state of the layer.
        :type state: Optional[Any]

        :return: The updated regularization loss.
        :rtype: torch.Tensor
        """
        next_V, next_a, next_Z = state
        self._regularization_loss += self.kwargs["spikes_regularization_factor"] * torch.sum(next_Z)
        # self._regularization_loss += 2e-6*torch.mean(torch.sum(next_Z, dim=-1)**2)
        return self._regularization_loss


class BellecLIFLayer(LIFLayer):
    """
    Layer implementing the LIF neuron model from the paper:
        "A solution to the learning dilemma for recurrent networks of spiking neurons"
        by Bellec et al. (2020) :cite:t:`bellec_solution_2020`.
    """

    def __init__(
            self,
            input_size: Optional[SizeTypes] = None,
            output_size: Optional[SizeTypes] = None,
            name: Optional[str] = None,
            use_recurrent_connection: bool = True,
            use_rec_eye_mask: bool = True,
            spike_func: Type[SpikeFunction] = HeavisidePhiApprox,
            dt: float = 1e-3,
            device: Optional[torch.device] = None,
            **kwargs
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            name=name,
            use_recurrent_connection=use_recurrent_connection,
            use_rec_eye_mask=use_rec_eye_mask,
            spike_func=spike_func,
            dt=dt,
            device=device,
            **kwargs
        )

    def forward(
            self,
            inputs: torch.Tensor,
            state: Tuple[torch.Tensor, ...] = None,
            **kwargs
    ):
        assert inputs.ndim == 2
        batch_size, nb_features = inputs.shape
        V, Z = self._init_forward_state(state, batch_size, inputs=inputs)
        input_current = torch.matmul(inputs, self.forward_weights)
        if self.use_recurrent_connection:
            rec_current = torch.matmul(Z, torch.mul(self.recurrent_weights, self.rec_mask))
        else:
            rec_current = 0.0
        next_V = (self.alpha * V + input_current + rec_current) - Z.detach() * self.threshold
        next_Z = self.spike_func.apply(next_V, self.threshold, self.gamma)
        return next_Z, (next_V, next_Z)


# @inherit_fields_docstring(fields=["Attributes"], bases=[BaseNeuronsLayer])
class IzhikevichLayer(BaseNeuronsLayer):
    """
    Izhikevich p.274

    Not usable for now, stay tuned.
    """

    def __init__(
            self,
            input_size: Optional[SizeTypes] = None,
            output_size: Optional[SizeTypes] = None,
            name: Optional[str] = None,
            use_recurrent_connection=True,
            use_rec_eye_mask=True,
            spike_func: Type[SpikeFunction] = HeavisideSigmoidApprox,
            dt=1e-3,
            device=None,
            **kwargs
    ):
        self.spike_func = spike_func
        super(IzhikevichLayer, self).__init__(
            input_size=input_size,
            output_size=output_size,
            name=name,
            use_recurrent_connection=use_recurrent_connection,
            use_rec_eye_mask=use_rec_eye_mask,
            dt=dt,
            device=device,
            **kwargs
        )

        self.C = torch.tensor(self.kwargs["C"], dtype=torch.float32, device=self._device)
        self.v_rest = torch.tensor(self.kwargs["v_rest"], dtype=torch.float32, device=self._device)
        self.v_th = torch.tensor(self.kwargs["v_th"], dtype=torch.float32, device=self._device)
        self.k = torch.tensor(self.kwargs["k"], dtype=torch.float32, device=self._device)
        self.a = torch.tensor(self.kwargs["a"], dtype=torch.float32, device=self._device)
        self.b = torch.tensor(self.kwargs["b"], dtype=torch.float32, device=self._device)
        self.c = torch.tensor(self.kwargs["c"], dtype=torch.float32, device=self._device)
        self.d = torch.tensor(self.kwargs["d"], dtype=torch.float32, device=self._device)
        self.v_peak = torch.tensor(self.kwargs["v_peak"], dtype=torch.float32, device=self._device)
        self.gamma = torch.tensor(self.kwargs["gamma"], dtype=torch.float32, device=self._device)
        self.initialize_weights_()

    def _set_default_kwargs(self):
        self.kwargs.setdefault("C", 100.0)
        self.kwargs.setdefault("v_rest", -60.0)
        self.kwargs.setdefault("v_th", -40.0)
        self.kwargs.setdefault("k", 0.7)
        self.kwargs.setdefault("a", 0.03)
        self.kwargs.setdefault("b", -2.0)
        self.kwargs.setdefault("c", -50.0)
        self.kwargs.setdefault("d", 100.0)
        self.kwargs.setdefault("v_peak", 35.0)
        if isinstance(self.spike_func, HeavisideSigmoidApprox):
            self.kwargs.setdefault("gamma", 100.0)
        else:
            self.kwargs.setdefault("gamma", 1.0)

    def initialize_weights_(self):
        super().initialize_weights_()
        gain = 1.0
        for param in self.parameters():
            if param.ndim > 2:
                torch.nn.init.xavier_normal_(param, gain=gain)
            else:
                torch.nn.init.normal_(param, std=gain)

    def create_empty_state(
            self,
            batch_size: int = 1,
            **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Create an empty state in the following form:
            ([membrane potential of shape (batch_size, self.output_size)],
            [membrane potential of shape (batch_size, self.output_size)],
            [spikes of shape (batch_size, self.output_size)])
        :param batch_size: The size of the current batch.
        :return: The current state.
        """
        V = self.v_rest * torch.ones(
            (batch_size, int(self._output_size)),
            device=self._device,
            dtype=torch.float32,
            requires_grad=True,
        )
        u = torch.zeros(
            (batch_size, int(self._output_size)),
            device=self._device,
            dtype=torch.float32,
            requires_grad=True,
        )
        Z = torch.zeros(
            (batch_size, int(self._output_size)),
            device=self._device,
            dtype=torch.float32,
            requires_grad=True,
        )
        return V, u, Z

    def forward(
            self,
            inputs: torch.Tensor,
            state: Tuple[torch.Tensor, ...] = None,
            **kwargs
    ):
        assert inputs.ndim == 2
        batch_size, nb_features = inputs.shape
        V, u, Z = self._init_forward_state(state, batch_size)
        input_current = torch.matmul(inputs, self.forward_weights)
        if self.use_recurrent_connection:
            rec_current = torch.matmul(Z, torch.mul(self.recurrent_weights, self.rec_mask))
        else:
            rec_current = 0.0
        is_reset = Z.detach()
        I = input_current + rec_current
        dVdt = self.k * (V - self.v_rest) * (V - self.v_th) - u + I
        next_V = (V + self.dt * dVdt / self.C) * (1.0 - is_reset) + self.c * is_reset
        dudt = self.a * (self.b * (V - self.v_rest) - u)
        next_u = (u + self.dt * dudt) + self.d * is_reset
        next_Z = self.spike_func.apply(next_V, self.v_peak, self.gamma)
        return next_Z, (next_V, next_u, next_Z)
