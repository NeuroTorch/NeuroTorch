import enum
from typing import Any, List, Optional, Sized, Tuple, Type, Union, Iterable

import numpy as np
import torch
from torch import nn

from ..base import SizedModule
from ...dimension import Dimension, DimensionProperty, DimensionsLike, SizeTypes
from ...transforms import to_tensor, ToDevice
from ...utils import format_pseudo_rn_seed, recursive_detach, ConnectivityConvention


class BaseLayer(SizedModule):
    """
    Base class for all layers.

    :Attributes:
        - :attr:`input_size` (Optional[Dimension]): The input size of the layer.
        - :attr:`output_size` (Optional[Dimension]): The output size of the layer.
        - :attr:`name` (str): The name of the layer.
        - :attr:`kwargs` (dict): Additional keyword arguments.

    """

    def __init__(
            self,
            input_size: Optional[SizeTypes] = None,
            output_size: Optional[SizeTypes] = None,
            name: Optional[str] = None,
            device: Optional[torch.device] = None,
            **kwargs
    ):
        """
        Constructor of the BaseLayer class.

        :param input_size: The input size of the layer.
        :type input_size: Optional[SizeTypes]
        :param output_size: The output size of the layer.
        :type output_size: Optional[SizeTypes]
        :param name: The name of the layer.
        :type name: Optional[str]
        :param device: The device of the layer. Defaults to the current available device.
        :type device: Optional[torch.device]
        :param kwargs: Additional keyword arguments.

        :keyword bool regularize: Whether to regularize the layer. If True, the method `update_regularization_loss`
            will be called after each forward pass. Defaults to False.
        :keyword bool freeze_weights: Whether to freeze the weights of the layer. Defaults to False.
        """
        super(BaseLayer, self).__init__(input_size=input_size, output_size=output_size, name=name)
        self._is_built = False
        self._freeze_weights = kwargs.get("freeze_weights", False)
        self._device = device
        if self._device is None:
            self._set_default_device_()
        self._device_transform = ToDevice(self.device)

        self.kwargs = kwargs
        self._set_default_kwargs()

        self.input_size = input_size
        self.output_size = output_size

        self._regularization_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

    @property
    def freeze_weights(self) -> bool:
        return self._freeze_weights

    @freeze_weights.setter
    def freeze_weights(self, freeze_weights: bool):
        self._freeze_weights = freeze_weights
        self.requires_grad_(self.requires_grad)

    @property
    def requires_grad(self):
        return not self.freeze_weights

    @property
    def is_ready_to_build(self) -> bool:
        return all(
            [
                s is not None
                for s in [
                self._input_size,
                (self._output_size if hasattr(self, "_output_size") else None)
            ]
            ]
        )

    @property
    def is_built(self) -> bool:
        return self._is_built

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        """
        Set the device of the layer and move all the parameters to the new device.

        :param device: The device to set.
        :type device: torch.device

        :return: None
        """
        self.to(device, non_blocking=True)

    def to(self, device: torch.device, non_blocking: bool = True, *args, **kwargs):
        """
        Move all the parameters of the layer to the specified device.

        :param device: The device to move the parameters to.
        :type device: torch.device
        :param non_blocking: Whether to move the parameters in a non-blocking way.
        :type non_blocking: bool
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.

        :return: self
        """
        self._device = device
        for module in self.modules():
            if module is not self and getattr(module, "device", device).type != device.type:
                module.to(device, non_blocking=non_blocking)
        return super(BaseLayer, self).to(*args, **kwargs)

    def __repr__(self):
        _repr = f"{self.__class__.__name__}"
        if self.name_is_set:
            _repr += f"<{self.name}>"
        _repr += f"({int(self.input_size)}->{int(self.output_size)}"
        _repr += f"{self.extra_repr()})"
        _repr += f"@{self.device}"
        return _repr

    def _format_size(self, size: Optional[SizeTypes], **kwargs) -> Optional[Dimension]:
        kwargs["filter_time"] = True
        return super(BaseLayer, self)._format_size(size, **kwargs)

    def _set_default_kwargs(self):
        pass

    def _set_default_device_(self):
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def build(self) -> 'BaseLayer':
        """
        Build the layer. This method must be call after the layer is initialized to make sure that the layer is ready
        to be used e.g. the input and output size is set, the weights are initialized, etc.

        :return: The layer itself.
        :rtype: BaseLayer
        """
        if self._is_built:
            raise ValueError("The layer can't be built multiple times.")
        if not self.is_ready_to_build:
            raise ValueError("Input size and output size must be specified before the build call.")
        self._is_built = True
        self.reset_regularization_loss()
        return self

    def create_empty_state(self, batch_size: int = 1, **kwargs) -> Tuple[torch.Tensor, ...]:
        """
        Create an empty state for the layer. This method must be implemented by the child class.

        :param batch_size: The batch size of the state.
        :type batch_size: int

        :return: The empty state.
        :rtype: Tuple[torch.Tensor, ...]
        """
        raise NotImplementedError()

    def _init_forward_state(
            self,
            state: Tuple[torch.Tensor, ...] = None,
            batch_size: int = 1,
            **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        if state is None:
            state = self.create_empty_state(batch_size, **kwargs)
        elif isinstance(state, (list, tuple)) and any([e is None for e in state]):
            empty_state = self.create_empty_state(batch_size, **kwargs)
            state = list(state)
            for i, e in enumerate(state):
                if e is None:
                    state[i] = empty_state[i]
            state = tuple(state)
        return state

    def infer_sizes_from_inputs(self, inputs: torch.Tensor):
        """
        Try to infer the input and output size of the layer from the inputs.

        :param inputs: The inputs to infer the size from.
        :type inputs: torch.Tensor

        :return: None
        """
        self.input_size = inputs.shape[-1]
        if self.output_size is None:
            raise ValueError("output_size must be specified before the forward call.")

    def __call__(self, inputs: torch.Tensor, state: torch.Tensor = None, *args, **kwargs):
        """
        Call the forward method of the layer. If the layer is not built, it will be built automatically.
        In addition, if :attr:`kwargs['regularize']` is set to True, the :meth: `update_regularization_loss` method
        will be called.

        :param inputs: The inputs to the layer.
        :type inputs: torch.Tensor

        :param args: The positional arguments to the forward method.
        :param kwargs: The keyword arguments to the forward method.

        :return: The output of the layer.
        """
        inputs, state = self._device_transform(inputs), self._device_transform(state)
        if not self.is_built:
            if not self.is_ready_to_build:
                self.infer_sizes_from_inputs(inputs)
            self.build()
        call_output = super(BaseLayer, self).__call__(inputs, state, *args, **kwargs)

        if isinstance(call_output, torch.Tensor):
            hidden_state = None
        elif isinstance(call_output, (List, Tuple)) and len(call_output) == 2:
            hidden_state = call_output[1]
        else:
            raise ValueError(
                "The forward method must return a torch.Tensor (the output of the layer) "
                "or a tuple of torch.Tensor (the output of the layer and the hidden state)."
            )
        if self.kwargs.get("regularize", False):
            self.update_regularization_loss(hidden_state)
        return call_output

    def forward(
            self,
            inputs: torch.Tensor,
            state: torch.Tensor = None,
            **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError()

    def initialize_weights_(self):
        """
        Initialize the weights of the layer. This method must be implemented by the child class.

        :return: None
        """
        pass

    def update_regularization_loss(self, state: Optional[Any] = None, *args, **kwargs) -> torch.Tensor:
        """
        Update the regularization loss for this layer. Each update call increments the regularization loss so at the end
        the regularization loss will be the sum of all calls to this function. This method is called at the end of each
        forward call automatically by the BaseLayer class.

        :param state: The current state of the layer.
        :type state: Optional[Any]
        :param args: Other positional arguments.
        :param kwargs: Other keyword arguments.

        :return: The updated regularization loss.
        :rtype: torch.Tensor
        """
        return self._regularization_loss

    def reset_regularization_loss(self):
        """
        Reset the regularization loss to zero.

        :return: None
        """
        self._regularization_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

    def get_and_reset_regularization_loss(self):
        """
        Get and reset the regularization loss for this layer. The regularization loss will be reset by the
        reset_regularization_loss method after it is returned.

        WARNING: If this method is not called after an integration, the update of the regularization loss can cause a
        memory leak. TODO: fix this.

        :return: The regularization loss.
        """
        loss = self.get_regularization_loss()
        self.reset_regularization_loss()
        return loss

    def get_regularization_loss(self) -> torch.Tensor:
        """
        Get the regularization loss for this layer.

        :return: The regularization loss.
        """
        return self._regularization_loss


# @inherit_fields_docstring(fields=["Attributes"], bases=[BaseLayer])
class BaseNeuronsLayer(BaseLayer):
    """
    A base class for layers that have neurons. This class provides two importants Parameters: the
    :attr:`forward_weights` and the :attr:`recurrent_weights`. Child classes must implement the :method:`forward`
    method and the :mth:`create_empty_state` method.

    :Attributes:
        - :attr:`forward_weights` (torch.nn.Parameter): The weights used to compute the output of the layer.
        - :attr:`recurrent_weights` (torch.nn.Parameter): The weights used to compute the hidden state of the layer.
        - :attr:`dt` (float): The time step of the layer.
        - :attr:`use_rec_eye_mask` (torch.Tensor): Whether to use the recurrent eye mask.
        - :attr:`rec_mask` (torch.Tensor): The recurrent eye mask.
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
        Initialize the layer.; See the :class:`BaseLayer` class for more details.;

        :param input_size: The input size of the layer;
        :type input_size: Optional[SizeTypes]
        :param output_size: The output size of the layer.
        :type output_size: Optional[SizeTypes]
        :param name: The name of the layer.
        :type name: Optional[str]
        :param use_recurrent_connection: Whether to use a recurrent connection. Default is True.
        :type use_recurrent_connection: bool
        :param use_rec_eye_mask: Whether to use a recurrent eye mask. Default is False. This mask will be used to
            mask to zero the diagonal of the recurrent connection matrix.
        :type use_rec_eye_mask: bool
        :param dt: The time step of the layer. Default is 1e-3.
        :type dt: float
        :param kwargs: Other keyword arguments.

        :keyword bool regularize: Whether to regularize the layer. If True, the method `update_regularization_loss` will
            be called after each forward pass. Defaults to False.
        :keyword str hh_init: The initialization method for the hidden state. Defaults to "zeros".
        :keyword float hh_init_mu: The mean of the hidden state initialization when hh_init is random . Defaults to 0.0.
        :keyword float hh_init_std: The standard deviation of the hidden state initialization when hh_init is random. Defaults to 1.0.
        :keyword int hh_init_seed: The seed of the hidden state initialization when hh_init is random. Defaults to 0.
        :keyword bool force_dale_law: Whether to force the Dale's law in the layer's weights. Defaults to False.
        :keyword Union[torch.Tensor, float] forward_sign: If force_dale_law is True, this parameter will be used to
            initialize the forward_sign vector. If it is a float, the forward_sign vector will be initialized with this
            value as the ration of inhibitory neurons. If it is a tensor, it will be used as the forward_sign vector.
        :keyword Union[torch.Tensor, float] recurrent_sign: If force_dale_law is True, this parameter will be used to
            initialize the recurrent_sign vector. If it is a float, the recurrent_sign vector will be initialized with
            this value as the ration of inhibitory neurons. If it is a tensor, it will be used as the recurrent_sign vector.
        :keyword Callable sign_activation: The activation function used to compute the sign of the weights i.e. the
            forward_sign and recurrent_sign vectors. Defaults to torch.nn.Tanh.
        """
        self.dt = dt
        self.use_recurrent_connection = use_recurrent_connection
        self._forward_weights = None
        self._forward_sign = None
        self.use_rec_eye_mask = use_rec_eye_mask
        self._recurrent_weights = None
        self._recurrent_sign = None
        self.rec_mask = None
        self._force_dale_law = kwargs.get("force_dale_law", False)
        self._connectivity_convention = ConnectivityConvention.from_other(
            kwargs.get("connectivity_convention", ConnectivityConvention.ItoJ)
        )
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            name=name,
            device=device,
            **kwargs
        )
        self.sign_activation = self.kwargs.get("sign_activation", torch.nn.Tanh())
        self.activation = self._init_activation(self.kwargs.get("activation", "identity"))

    @property
    def forward_weights(self) -> torch.nn.Parameter:
        """
        Get the forward weights.

        :return: The forward weights.
        """
        if self.force_dale_law:
            return torch.pow(self._forward_weights, 2) * self.forward_sign
        return self._forward_weights

    @forward_weights.setter
    def forward_weights(self, value: torch.nn.Parameter):
        """
        Set the forward weights.

        :param value: The forward weights.
        """
        if not isinstance(value, torch.nn.Parameter):
            value = torch.nn.Parameter(to_tensor(value), requires_grad=self.requires_grad)
        self._forward_weights = value

    @property
    def recurrent_weights(self) -> torch.nn.Parameter:
        """
        Get the recurrent weights.

        :return: The recurrent weights.
        """
        if self.force_dale_law:
            return torch.pow(self._recurrent_weights, 2) * self.recurrent_sign
        return self._recurrent_weights

    @recurrent_weights.setter
    def recurrent_weights(self, value: torch.nn.Parameter):
        """
        Set the recurrent weights.

        :param value: The recurrent weights.
        """
        if not isinstance(value, torch.nn.Parameter):
            value = torch.nn.Parameter(to_tensor(value), requires_grad=self.requires_grad)
        self._recurrent_weights = value

    @property
    def force_dale_law(self) -> bool:
        """
        Get whether to force the Dale's law.

        :return: Whether to force the Dale's law.
        """
        return self._force_dale_law

    @property
    def forward_sign(self) -> Optional[torch.nn.Parameter]:
        """
        Get the forward sign.

        :return: The forward sign.
        """
        if self._forward_sign is None:
            return None
        return self.sign_activation(self._forward_sign)

    @forward_sign.setter
    def forward_sign(self, value: torch.nn.Parameter):
        """
        Set the forward sign.

        :param value: The forward sign.
        """
        if not isinstance(value, torch.nn.Parameter):
            value = torch.nn.Parameter(value, requires_grad=self.force_dale_law and self.requires_grad)
        self._forward_sign = value

    @property
    def recurrent_sign(self) -> Optional[torch.nn.Parameter]:
        """
        Get the recurrent sign.

        :return: The recurrent sign.
        """
        if self._recurrent_sign is None:
            return None
        return self.sign_activation(self._recurrent_sign)

    @recurrent_sign.setter
    def recurrent_sign(self, value: torch.nn.Parameter):
        """
        Set the recurrent sign.

        :param value: The recurrent sign.
        """
        if not isinstance(value, torch.nn.Parameter):
            value = torch.nn.Parameter(value, requires_grad=self.force_dale_law)
        self._recurrent_sign = value

    @property
    def connectivity_convention(self) -> ConnectivityConvention:
        """
        Get the connectivity convention.

        :return: The connectivity convention.
        """
        return self._connectivity_convention

    def get_forward_weights_parameter(self) -> torch.nn.Parameter:
        """
        Get the forward weights parameter.

        :return: The forward weights parameter.
        """
        return self._forward_weights

    def set_forward_weights_parameter(self, parameter: torch.nn.Parameter):
        """
        Set the forward weights parameter.

        :param parameter: The forward weights parameter.
        """
        self._forward_weights = parameter

    def get_recurrent_weights_parameter(self) -> torch.nn.Parameter:
        """
        Get the recurrent weights parameter.

        :return: The recurrent weights parameter.
        """
        return self._recurrent_weights

    def set_recurrent_weights_parameter(self, parameter: torch.nn.Parameter):
        """
        Set the recurrent weights parameter.

        :param parameter: The recurrent weights parameter.
        """
        self._recurrent_weights = parameter

    def get_forward_sign_parameter(self) -> torch.nn.Parameter:
        """
        Get the forward sign parameter.

        :return: The forward sign parameter.
        """
        return self._forward_sign

    def set_forward_sign_parameter(self, parameter: torch.nn.Parameter):
        """
        Set the forward sign parameter.

        :param parameter: The forward sign parameter.
        """
        self._forward_sign = parameter

    def get_recurrent_sign_parameter(self) -> torch.nn.Parameter:
        """
        Get the recurrent sign parameter.

        :return: The recurrent sign parameter.
        """
        return self._recurrent_sign

    def set_recurrent_sign_parameter(self, parameter: torch.nn.Parameter):
        """
        Set the recurrent sign parameter.

        :param parameter: The recurrent sign parameter.
        """
        self._recurrent_sign = parameter

    def get_forward_weights_data(self) -> torch.Tensor:
        """
        Get the forward weights data.

        :return: The forward weights data.
        """
        return self._forward_weights.data

    def set_forward_weights_data(self, data: torch.Tensor):
        """
        Set the forward weights data.

        :param data: The forward weights data.
        """
        self._forward_weights.data = to_tensor(data).to(self.device)

    def get_recurrent_weights_data(self) -> torch.Tensor:
        """
        Get the recurrent weights data.

        :return: The recurrent weights data.
        """
        return self._recurrent_weights.data

    def set_recurrent_weights_data(self, data: torch.Tensor):
        """
        Set the recurrent weights data.

        :param data: The recurrent weights data.
        """
        self._recurrent_weights.data = to_tensor(data).to(self.device)

    def get_weights_parameters(self) -> List[torch.nn.Parameter]:
        """
        Get the weights parameters.

        :return: The weights parameters.
        """
        parameters = [self._forward_weights]
        if self.use_recurrent_connection:
            parameters.append(self._recurrent_weights)
        return parameters

    def get_sign_parameters(self) -> List[torch.nn.Parameter]:
        """
        Get the sign parameters.

        :return: The sign parameters.
        """
        parameters = []
        if self.force_dale_law:
            parameters.append(self._forward_sign)
            if self.use_recurrent_connection:
                parameters.append(self._recurrent_sign)
        return parameters

    def create_empty_state(
            self,
            batch_size: int = 1,
            **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        self.kwargs.setdefault("hh_init", "zeros")
        self.kwargs.setdefault("hh_init_mu", 0.0)
        self.kwargs.setdefault("hh_init_std", 1.0)

        n_hh = kwargs.get("n_hh", 1)
        if self.kwargs["hh_init"] == "zeros":
            state = tuple(
                [torch.zeros(
                    (batch_size, int(self.output_size)),
                    device=self.device,
                    dtype=torch.float32,
                    requires_grad=True,
                ) for _ in range(n_hh)]
            )
        elif self.kwargs["hh_init"] == "random":
            mu, std = self.kwargs.get("hh_init_mu", 0.0), self.kwargs.get("hh_init_std", 1.0)
            gen = torch.Generator(device=self.device)
            gen.manual_seed(format_pseudo_rn_seed(self.kwargs.get("hh_init_seed", None)))
            state = [(torch.rand(
                (batch_size, int(self.output_size)),
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
                generator=gen,
            ) * std + mu) for _ in range(n_hh)]
        elif self.kwargs["hh_init"] == "inputs":
            assert "inputs" in kwargs, "inputs must be provided to initialize the state"
            assert kwargs["inputs"].shape == (batch_size, int(self.output_size))
            state = [kwargs["inputs"].clone() for _ in range(n_hh)]
        elif self.kwargs["hh_init"].lower() == "given":
            assert "h0" in self.kwargs, "h0 must be provided as a tuple of tensors when hh_init is 'given'."
            h0 = self.kwargs["h0"]
            assert isinstance(h0, (tuple, list)), "h0 must be a tuple of tensors."
            state = [to_tensor(h0_, dtype=torch.float32).to(self.device) for h0_ in h0]
        else:
            raise ValueError("Hidden state init method not known. Please use 'zeros', 'inputs', 'random' or 'given'.")
        return tuple(state)

    def forward(
            self,
            inputs: torch.Tensor,
            state: torch.Tensor = None,
            **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError()

    def _init_forward_sign_(self):
        if self.kwargs.get("forward_sign", None) is None:
            self.kwargs.pop("forward_sign", None)
        if "forward_sign" in self.kwargs and self.force_dale_law:
            if isinstance(self.kwargs["forward_sign"], float):
                assert 0.0 <= self.kwargs["forward_sign"] <= 1.0, "forward_sign must be in [0, 1]"
                n_inh = int(int(self.input_size) * self.kwargs["forward_sign"])
                inh_indexes = torch.randperm(int(self.input_size))[:n_inh]
                self.kwargs["forward_sign"] = np.abs(np.random.normal(size=(int(self.input_size), 1)))
                self.kwargs["forward_sign"][inh_indexes] *= -1
            assert self.kwargs["forward_sign"].shape == (int(self.input_size), 1), \
                "forward_sign must be a float or a tensor of shape (input_size, 1)"
            self._forward_sign.data = to_tensor(self.kwargs["forward_sign"]).to(self.device)
            with torch.no_grad():
                self._forward_weights.data = torch.sqrt(torch.abs(self._forward_weights.data))
        elif self.force_dale_law:
            torch.nn.init.normal_(self._forward_sign)

    def _init_recurrent_sign_(self):
        if self.kwargs.get("recurrent_sign", None) is None:
            self.kwargs.pop("recurrent_sign", None)
        if "recurrent_sign" in self.kwargs and self.force_dale_law and self.use_recurrent_connection:
            if isinstance(self.kwargs["recurrent_sign"], float):
                assert 0.0 <= self.kwargs["recurrent_sign"] <= 1.0, "recurrent_sign must be in [0, 1]"
                n_inh = int(int(self.output_size) * self.kwargs["recurrent_sign"])
                inh_indexes = torch.randperm(int(self.output_size))[:n_inh]
                self.kwargs["recurrent_sign"] = np.abs(np.random.normal(size=(int(self.output_size), 1)))
                self.kwargs["recurrent_sign"][inh_indexes] *= -1
            assert self.kwargs["recurrent_sign"].shape == (int(self.output_size), 1), \
                "recurrent_sign must be a float or a tensor of shape (output_size, 1)"
            self._recurrent_sign.data = to_tensor(self.kwargs["recurrent_sign"]).to(self.device)
            with torch.no_grad():
                self._recurrent_weights.data = torch.sqrt(torch.abs(self._recurrent_weights.data))
        elif self.force_dale_law and self.use_recurrent_connection:
            torch.nn.init.xavier_normal_(self._recurrent_sign)

    def _init_activation(self, activation: Union[torch.nn.Module, str] = "identity"):
        """
        Initialise the activation function.

        :param activation: Activation function.
        :type activation: Union[torch.nn.Module, str]
        """
        str_to_activation = {
            "identity": torch.nn.Identity(),
            "relu"    : torch.nn.ReLU(),
            "tanh"    : torch.nn.Tanh(),
            "sigmoid" : torch.nn.Sigmoid(),
            "softmax" : torch.nn.Softmax(dim=-1),
            "elu"     : torch.nn.ELU(),
            "selu"    : torch.nn.SELU(),
            "prelu"   : torch.nn.PReLU(),
            "leakyrelu": torch.nn.LeakyReLU(),
            "leaky_relu": torch.nn.LeakyReLU(),
            "logsigmoid": torch.nn.LogSigmoid(),
            "log_sigmoid": torch.nn.LogSigmoid(),
            "logsoftmax": torch.nn.LogSoftmax(dim=-1),
            "log_softmax": torch.nn.LogSoftmax(dim=-1),
        }
        if isinstance(activation, str):
            if activation.lower() not in str_to_activation.keys():
                raise ValueError(
                    f"Activation {activation} is not implemented. Please use one of the following: "
                    f"{str_to_activation.keys()} or provide a torch.nn.Module."
                )
            self.activation = str_to_activation[activation.lower()]
        else:
            self.activation = activation
        return self.activation

    def initialize_weights_(self):
        super().initialize_weights_()
        if self.kwargs.get("forward_weights", None) is not None:
            self._forward_weights.data = to_tensor(self.kwargs["forward_weights"]).to(self.device)
        else:
            torch.nn.init.xavier_normal_(self._forward_weights)

        if self.kwargs.get("recurrent_weights", None) is not None and self.use_recurrent_connection:
            self._recurrent_weights.data = to_tensor(self.kwargs["recurrent_weights"]).to(self.device)
        elif self.use_recurrent_connection:
            torch.nn.init.xavier_normal_(self._recurrent_weights)

        self._init_forward_sign_()
        self._init_recurrent_sign_()

    def build(self) -> 'BaseNeuronsLayer':
        """
        Build the layer. This method must be call after the layer is initialized to make sure that the layer is ready
        to be used e.g. the input and output size is set, the weights are initialized, etc.

        In this method the :attr:`forward_weights`, :attr:`recurrent_weights` and :attr: `rec_mask` are created and
        finally the method :meth:`initialize_weights_` is called.

        :return: The layer itself.
        :rtype: BaseLayer
        """
        super().build()
        self._forward_weights = nn.Parameter(
            torch.empty((int(self.input_size), int(self.output_size)), device=self.device, dtype=torch.float32),
            requires_grad=self.requires_grad
        )
        if self.force_dale_law:
            self._forward_sign = torch.nn.Parameter(
                torch.empty((int(self.input_size), 1), dtype=torch.float32, device=self.device),
                requires_grad=self.force_dale_law
            )

        if self.use_recurrent_connection:
            self._recurrent_weights = nn.Parameter(
                torch.empty((int(self.output_size), int(self.output_size)), device=self.device, dtype=torch.float32),
                requires_grad=self.requires_grad
            )
            if self.use_rec_eye_mask:
                self.rec_mask = nn.Parameter(
                    (1 - torch.eye(int(self.output_size), device=self.device, dtype=torch.float32)),
                    requires_grad=False
                )
            else:
                self.rec_mask = nn.Parameter(
                    torch.ones(
                        (int(self.output_size), int(self.output_size)), device=self.device, dtype=torch.float32
                    ),
                    requires_grad=False
                )
            if self.force_dale_law:
                self._recurrent_sign = torch.nn.Parameter(
                    torch.empty((int(self.output_size), 1), dtype=torch.float32, device=self.device),
                    requires_grad=self.force_dale_law
                )
        self.initialize_weights_()
        return self

    def __repr__(self):
        _repr = f"{self.__class__.__name__}"
        if self.name_is_set:
            _repr += f"<{self.name}>"
        if self.force_dale_law:
            _repr += f"[Dale]"
        _repr += f"({int(self.input_size)}"
        if self.use_recurrent_connection:
            _repr += "<"
        _repr += f"->{int(self.output_size)}"
        _repr += f"{self.extra_repr()}"
        _repr += f", activation:{self.activation}"
        if self.force_dale_law:
            _repr += f", sign_activation:{self.sign_activation}"
        _repr += ")"
        if self.freeze_weights:
            _repr += "[frozen]"
        _repr += f"@{self.device}"
        return _repr

