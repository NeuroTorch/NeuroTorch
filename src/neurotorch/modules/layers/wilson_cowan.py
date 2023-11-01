from typing import Optional, Tuple, Union

import torch

from .base import BaseNeuronsLayer
from ...dimension import SizeTypes
from ...transforms import to_tensor


# @inherit_fields_docstring(fields=["Attributes"], bases=[BaseNeuronsLayer])
class WilsonCowanLayer(BaseNeuronsLayer):
    """
    This layer is use for Wilson-Cowan neuronal dynamics.
    This dynamic is also referred to as firing rate model.
    Wilson-Cowan dynamic is great for neuronal calcium activity.
    This layer use recurrent neural network (RNN).
    The number of parameters that are trained is N^2 (+2N if mu and r is train)
    where N is the number of neurons.

    For references, please read:

        - Excitatory and Inhibitory Interactions in Localized Populations of Model Neurons :cite:t:`wilson1972excitatory`
        - Beyond Wilson-Cowan dynamics: oscillations and chaos without inhibitions :cite:t:`PainchaudDoyonDesrosiers2022`
        - Neural Network dynamic :cite:t:`VogelsTimRajanAbbott2005NeuralNetworkDynamics`.

    The Wilson-Cowan dynamic is one of many dynamical models that can be used
    to model neuronal activity. To explore more continuous and Non-linear dynamics,
    please read Nonlinear Neural Network: Principles, Mechanisms, and Architecture :cite:t:`GROSSBERG198817`.


    """

    def __init__(
            self,
            input_size: Optional[SizeTypes] = None,
            output_size: Optional[SizeTypes] = None,
            dt: float = 1e-3,
            use_recurrent_connection: bool = False,
            device=None,
            **kwargs
    ):
        """
        :param input_size: size of the input
        :type input_size: Optional[SizeTypes]
        :param output_size: size of the output
            If we are predicting time series -> input_size = output_size
        :type output_size: Optional[SizeTypes]
        :param learning_type: Type of learning for the gradient descent
        :type learning_type: LearningType
        :param dt: Time step (Euler's discretisation)
        :type dt: float
        :param device: device for computation
        :type device: torch.device
        :param kwargs: Additional parameters for the Wilson-Cowan dynamic.

        :keyword Union[torch.Tensor, np.ndarray] forward_weights: Forward weights of the layer.
        :keyword float std_weight: Instability of the initial random matrix.
        :keyword Union[float, torch.Tensor] mu: Activation threshold. If torch.Tensor -> shape (1, number of neurons).
        :keyword float mean_mu: Mean of the activation threshold (if learn_mu is True).
        :keyword float std_mu: Standard deviation of the activation threshold (if learn_mu is True).
        :keyword bool learn_mu: Whether to train the activation threshold.
        :keyword float tau: Decay constant of RNN unit.
        :keyword bool learn_tau: Whether to train the decay constant.
        :keyword float r: Transition rate of the RNN unit. If torch.Tensor -> shape (1, number of neurons).
        :keyword float mean_r: Mean of the transition rate (if learn_r is True).
        :keyword float std_r: Standard deviation of the transition rate (if learn_r is True).
        :keyword bool learn_r: Whether to train the transition rate.

        Remarks: Parameter mu and r can only be a parameter as a vector.
        """
        super(WilsonCowanLayer, self).__init__(
            input_size=input_size,
            output_size=output_size,
            use_recurrent_connection=use_recurrent_connection,
            dt=dt,
            device=device,
            **kwargs
        )
        self.std_weight = self.kwargs["std_weight"]
        self.mu = torch.nn.Parameter(to_tensor(self.kwargs["mu"]).to(self.device), requires_grad=False)
        self.mean_mu = self.kwargs["mean_mu"]
        self.std_mu = self.kwargs["std_mu"]
        self.learn_mu = self.kwargs["learn_mu"]
        self.tau_sqrt = torch.nn.Parameter(
            torch.sqrt(to_tensor(self.kwargs["tau"])).to(self.device), requires_grad=False
        )
        self.learn_tau = self.kwargs["learn_tau"]
        self.r_sqrt = torch.nn.Parameter(
            torch.sqrt(to_tensor(self.kwargs["r"], dtype=torch.float32)).to(self.device), requires_grad=False
        )
        self.mean_r = self.kwargs["mean_r"]
        self.std_r = self.kwargs["std_r"]
        self.learn_r = self.kwargs["learn_r"]
        self.activation = self._init_activation(self.kwargs["activation"])

    def _set_default_kwargs(self):
        self.kwargs.setdefault("std_weight", 1.0)
        self.kwargs.setdefault("mu", 0.0)
        self.kwargs.setdefault("tau", 1.0)
        self.kwargs.setdefault("learn_tau", False)
        self.kwargs.setdefault("learn_mu", False)
        self.kwargs.setdefault("mean_mu", 2.0)
        self.kwargs.setdefault("std_mu", 0.0)
        self.kwargs.setdefault("r", 0.0)
        self.kwargs.setdefault("learn_r", False)
        self.kwargs.setdefault("mean_r", 2.0)
        self.kwargs.setdefault("std_r", 0.0)
        self.kwargs.setdefault("hh_init", "inputs")
        self.kwargs.setdefault("activation", "sigmoid")

    def _assert_kwargs(self):
        assert self.std_weight >= 0.0, "std_weight must be greater or equal to 0.0"
        assert self.std_mu >= 0.0, "std_mu must be greater or equal to 0.0"
        assert self.tau > self.dt, "tau must be greater than dt"

    @property
    def r(self):
        """
        This property is used to ensure that the transition rate will never be negative if trained.
        """
        return torch.pow(self.r_sqrt, 2)

    @r.setter
    def r(self, value):
        self.r_sqrt.data = torch.sqrt(torch.abs(to_tensor(value, dtype=torch.float32))).to(self.device)

    @property
    def tau(self):
        """
        This property is used to ensure that the decay constant will never be negative if trained.
        """
        return torch.pow(self.tau_sqrt, 2)

    @tau.setter
    def tau(self, value):
        self.tau_sqrt.data = torch.sqrt(torch.abs(to_tensor(value, dtype=torch.float32))).to(self.device)

    def initialize_weights_(self):
        """
        Initialize the parameters (weights) that will be trained.
        """
        super().initialize_weights_()
        if self.kwargs.get("forward_weights", None) is not None:
            self._forward_weights.data = to_tensor(self.kwargs["forward_weights"]).to(self.device)
        else:
            torch.nn.init.normal_(self._forward_weights, mean=0.0, std=self.std_weight)

        # If mu is not a parameter, it takes the value 0.0 unless stated otherwise by user
        # If mu is a parameter, it is initialized as a vector with the correct mean and std
        # unless stated otherwise by user.
        if self.learn_mu:
            if self.mu.dim() == 0:  # if mu is a scalar and a parameter -> convert it to a vector
                self.mu.data = torch.empty((1, int(self.output_size)), dtype=torch.float32, device=self.device)
            self.mu = torch.nn.Parameter(self.mu, requires_grad=self.requires_grad)
            torch.nn.init.normal_(self.mu, mean=self.mean_mu, std=self.std_mu)
        if self.learn_r:
            _r = torch.empty((1, int(self.output_size)), dtype=torch.float32, device=self.device)
            torch.nn.init.normal_(_r, mean=self.mean_r, std=self.std_r)
            self.r_sqrt = torch.nn.Parameter(torch.sqrt(torch.abs(_r)), requires_grad=self.requires_grad)
        if self.learn_tau:
            self.tau_sqrt = torch.nn.Parameter(self.tau_sqrt, requires_grad=self.requires_grad)

    def create_empty_state(self, batch_size: int = 1, **kwargs) -> Tuple[torch.Tensor]:
        if self.kwargs["hh_init"] == "zeros":
            state = [torch.zeros(
                (batch_size, int(self.output_size)),
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            ) for _ in range(1)]
        elif self.kwargs["hh_init"] == "random":
            mu, std = self.kwargs.get("hh_init_mu", 0.0), self.kwargs.get("hh_init_std", 1.0)
            gen = torch.Generator(device=self.device)
            gen.manual_seed(self.kwargs.get("hh_init_seed", 0))
            state = [(torch.rand(
                (batch_size, int(self.output_size)),
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
                generator=gen,
            ) * std + mu) for _ in range(1)]
        elif self.kwargs["hh_init"] == "inputs":
            assert "inputs" in kwargs, "inputs must be provided to initialize the state"
            assert kwargs["inputs"].shape == (batch_size, int(self.output_size))
            state = (kwargs["inputs"].clone(),)
        else:
            state = super().create_empty_state(batch_size, **kwargs)
        return tuple(state)

    def forward(
            self,
            inputs: torch.Tensor,
            state: Optional[Tuple[torch.Tensor, ...]] = None,
            **kwargs
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Forward pass.
        With Euler discretisation, Wilson-Cowan equation becomes:

        output = input * (1 - dt/tau) + dt/tau * (1 - input @ r) * sigmoid(input @ forward_weight - mu)

        :param inputs: time series at a time t of shape (batch_size, number of neurons)
            Remark: if you use to compute a time series, use batch_size = 1.
        :type inputs: torch.Tensor
        :param state: State of the layer (only for SNN -> not use for RNN)
        :type state: Optional[Tuple[torch.Tensor, ...]]

        :return: (time series at a time t+1, State of the layer -> None)
        :rtype: Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]
        """
        batch_size, nb_features = inputs.shape
        hh, = self._init_forward_state(state, batch_size, inputs=inputs)
        ratio_dt_tau = self.dt / self.tau

        if self.use_recurrent_connection:
            rec_inputs = torch.matmul(hh, torch.mul(self.recurrent_weights, self.rec_mask))
        else:
            rec_inputs = 0.0

        transition_rate = (1 - hh * self.r)
        activation = self.activation(rec_inputs + torch.matmul(inputs, self.forward_weights) - self.mu)
        output = hh * (1 - ratio_dt_tau) + transition_rate * activation * ratio_dt_tau
        return output, (torch.clone(output),)


class WilsonCowanCURBDLayer(WilsonCowanLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_empty_state(self, batch_size: int = 1, **kwargs) -> Tuple[torch.Tensor, ...]:
        if self.kwargs["hh_init"].lower() == "given":
            assert "h0" in self.kwargs, "h0 must be provided as a tuple of tensors when hh_init is 'given'."
            h0 = self.kwargs["h0"]
            assert isinstance(h0, (tuple, list)), "h0 must be a tuple of tensors."
            state = [to_tensor(h0_, dtype=torch.float32).to(self.device) for h0_ in h0]
        else:
            state = super().create_empty_state(batch_size, **kwargs)
        return tuple(state)

    def forward(
            self,
            inputs: torch.Tensor,
            state: Optional[Tuple[torch.Tensor, ...]] = None,
            **kwargs
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        batch_size, nb_features = inputs.shape

        out_shape = tuple(inputs.shape[:-1]) + (self.forward_weights.shape[-1],)  # [*, f_out]
        inputs_view = inputs.view(-1, inputs.shape[-1])  # [*, f_in] -> [B, f_in]

        hh, = self._init_forward_state(state, batch_size, inputs=inputs_view, **kwargs)  # [B, f_out]
        post_activation = self.activation(hh)  # [B, f_out]

        if self.use_recurrent_connection:
            # [B, f_out] @ [f_out, f_out] -> [B, f_out]
            rec_inputs = torch.matmul(post_activation, torch.mul(self.recurrent_weights, self.rec_mask))
        else:
            rec_inputs = 0.0

        # [B, f_in] @ [f_in, f_out] -> [B, f_out]
        weighted_current = torch.matmul(inputs_view, self.forward_weights)
        jr = (rec_inputs + weighted_current)  # [B, f_out]
        next_hh = hh + self.dt * (-hh + jr) / self.tau  # [B, f_out]
        output = post_activation.view(out_shape)  # [B, f_out] -> [*, f_out]
        return output, (next_hh,)

