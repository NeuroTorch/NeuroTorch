import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Type, Union
from typing import OrderedDict as OrderedDictType

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import (
    BaseLayer,
    BaseModel,
    LayerType,
    LayerType2Layer,
    SpikeFuncType,
    SpikeFunction
)
from .base import NamedModule
from .wrappers import NamedModuleWrapper
from ..dimension import Dimension


Acceptable_Spike_Func = Union[Type[SpikeFunction], SpikeFuncType]
Acceptable_Spike_Funcs = Union[Acceptable_Spike_Func, Iterable[Acceptable_Spike_Func]]
Acceptable_Layer_Type = Union[Type[BaseLayer], LayerType]
Acceptable_Layer_Types = Union[Acceptable_Layer_Type, Iterable[Acceptable_Layer_Type]]
IntDimension = Union[int, Dimension]


class Sequential(BaseModel):
    """
    The Sequential is a neural network that is constructed by stacking layers.

    .. image:: ../../images/modules/Sequential_model_schm.png
        :width: 300
        :align: center

    :Attributes:
        - :attr:`input_layers` (torch.nn.ModuleDict): The input layers of the model.
        - :attr:`hidden_layers` (torch.nn.ModuleList): The hidden layers of the model.
        - :attr:`output_layers` (torch.nn.ModuleDict): The output layers of the model.
        - :attr:`foresight_time_steps` (int): The number of time steps that the model will forecast.
    """

    @staticmethod
    def _format_input_output_layers(
            layers: Iterable[Union[Iterable[torch.nn.Module], torch.nn.Module]],
            default_prefix_layer_name: str = "layer",
    ) -> OrderedDictType[str, NamedModule]:
        """
        Format the input or output layers. The format is an ordered dictionary of the form {layer_name: layer}.

        :param layers: The input or output layers.
        :type layers: Iterable[Union[Iterable[torch.nn.Module], torch.nn.Module]]
        :param default_prefix_layer_name: The default prefix of the layer name. The prefix is used when the name of
        the layer is not specified.
        :type default_prefix_layer_name: str

        :return: The formatted input or output layers.
        :rtype: OrderedDict[str, NamedModule]
        """
        layers: Iterable[torch.nn.Module] = [layers] if not isinstance(layers, (Iterable, Mapping)) else layers
        if isinstance(layers, Mapping):
            layers: OrderedDictType[str, NamedModule] = OrderedDict(
                (k, (v if isinstance(v, NamedModule) else NamedModuleWrapper(v))) for k, v in layers.items()
            )
            for layer_key, layer in layers.items():
                if not layer.name_is_set:
                    layer.name = layer_key
            assert all(layer_key == layer.name for layer_key, layer in layers.items()), \
                "The layer names must be the same as the keys."
        else:
            layers: Iterable[NamedModule] = [
                (layer if isinstance(layer, NamedModule) else NamedModuleWrapper(layer)) for layer in layers
            ]
            for layer_idx, layer in enumerate(layers):
                if not layer.name_is_set:
                    layer.name = f"{default_prefix_layer_name}_{layer_idx}"
            assert len([layer.name for layer in layers]) == len(set([layer.name for layer in layers])), \
                "There are layers with the same name. Please specify the names of the layers without duplicates."
            layers: OrderedDict[str, NamedModule] = OrderedDict((layer.name, layer) for layer in layers)
        return layers

    @staticmethod
    def _format_hidden_layers(
            layers: Iterable[torch.nn.Module],
            default_prefix_layer_name: str = "hidden",
    ) -> List[NamedModule]:
        """
        Format the hidden layers. The format is a list of the form [layer, ...].

        :param layers: The hidden layers.
        :type layers: Iterable[torch.nn.Module]
        :param default_prefix_layer_name: The default prefix of the layer name. The prefix is used when the name of
        the layer is not specified.
        :type default_prefix_layer_name: str

        :return: The formatted hidden layers.
        :rtype: List[NamedModule]
        """
        layers: Iterable[NamedModule] = [
            (layer if isinstance(layer, NamedModule) else NamedModuleWrapper(layer)) for layer in layers
        ]
        for i, layer in enumerate(layers):
            if not layer.name_is_set:
                layer.name = f"{default_prefix_layer_name}_{i}"
        return list(layers)

    @staticmethod
    def _format_layers(
            layers: Iterable[Union[Iterable[torch.nn.Module], torch.nn.Module]]
    ) -> Tuple[OrderedDict, List, OrderedDict]:
        """
        Format the given layers. The format is a tuple of the form:

        ::

            (
                OrderedDict({input_layer_name: input_layer}),
                List(hidden_layers),
                OrderedDict({output_layer_name: output_layer}),
            )

        :param layers: The layers.
        :type layers: Iterable[Union[Iterable[BaseLayer], BaseLayer]]

        :return: The formatted layers.
        :rtype: Tuple[OrderedDict, List, OrderedDict]
        """
        if not isinstance(layers, Iterable):
            layers = [layers]
        if len(layers) > 1:
            input_layers = Sequential._format_input_output_layers(layers[0], "input")
        else:
            input_layers = nn.ModuleDict()

        if len(layers) > 2:
            hidden_layers = layers[1:-1]
            if not isinstance(hidden_layers, Iterable):
                hidden_layers = [hidden_layers]
        else:
            hidden_layers = []
        hidden_layers = Sequential._format_hidden_layers(hidden_layers)

        output_layers = Sequential._format_input_output_layers(layers[-1], "output")
        return input_layers, hidden_layers, output_layers

    @staticmethod
    def _layers_containers_to_modules(
            inputs_layers: OrderedDict,
            hidden_layers: List,
            outputs_layers: OrderedDict
    ) -> Tuple[nn.ModuleDict, nn.ModuleList, nn.ModuleDict]:
        """
        Convert the input, hidden and output layers containers to modules.

        :param inputs_layers: The input layers.
        :type inputs_layers: OrderedDict
        :param hidden_layers: The hidden layers.
        :type hidden_layers: List
        :param outputs_layers: The output layers.
        :type outputs_layers: OrderedDict

        :return: The input, hidden and output layers modules.
        :rtype: Tuple[nn.ModuleDict, nn.ModuleList, nn.ModuleDict]
        """
        input_layers = nn.ModuleDict(inputs_layers)
        hidden_layers = nn.ModuleList(hidden_layers)
        output_layers = nn.ModuleDict(outputs_layers)
        return input_layers, hidden_layers, output_layers

    @staticmethod
    def _format_layer_type_(
            layer_type: Optional[Acceptable_Layer_Type]
    ) -> Optional[Type[BaseLayer]]:
        warnings.warn(
            "This function is not used anymore in the SequentialModel.",
            DeprecationWarning
        )
        if isinstance(layer_type, LayerType):
            layer_type = LayerType2Layer[layer_type]
        return layer_type

    @staticmethod
    def _format_hidden_neurons_(n_hidden_neurons: Optional[Union[int, Iterable[int]]]) -> List[int]:
        warnings.warn(
            "This function is not used anymore in the SequentialModel.",
            DeprecationWarning
        )
        if n_hidden_neurons is None:
            return []
        if not isinstance(n_hidden_neurons, Iterable):
            n_hidden_neurons = [n_hidden_neurons]
        return n_hidden_neurons

    def __init__(
            self,
            layers: Iterable[Union[Iterable[torch.nn.Module], torch.nn.Module]],
            name: str = "Sequential",
            checkpoint_folder: str = "checkpoints",
            device: Optional[torch.device] = None,
            input_transform: Optional[Union[Dict[str, Callable], List[Callable]]] = None,
            output_transform: Optional[Union[Dict[str, Callable], List[Callable]]] = None,
            **kwargs
    ):
        """
        The Sequential is a neural network that is constructed by stacking layers.

        :param layers: The layers to be used in the model. One of the following structure is expected:
        ::

            layers = [
                [*inputs_layers, ],
                *hidden_layers,
                [*output_layers, ]
            ]
            or
            layers = [
                input_layer,
                *hidden_layers,
                output_layer
            ]

        :type layers: Iterable[Union[Iterable[torch.nn.Module], torch.nn.Module]]
        :param name: The name of the model.
        :type name: str
        :param checkpoint_folder: The folder where the checkpoints are saved.
        :type checkpoint_folder: str
        :param device: The device to use.
        :type device: torch.device
        :param input_transform: The transform to apply to the input. The input_transform must work on a single datum.
        :type input_transform: Union[Dict[str, Callable], List[Callable]]
        :param output_transform: The transform to apply to the output trace. The output_transform must work batch-wise.
        :type output_transform: Union[Dict[str, Callable], List[Callable]]
        :param kwargs: Additional keyword arguments.
        """
        input_layers, hidden_layers, output_layers = self._format_layers(layers)
        self._ordered_inputs_names = [layer.name for _, layer in input_layers.items()]
        self._ordered_outputs_names = [layer.name for _, layer in output_layers.items()]
        super(Sequential, self).__init__(
            # TODO: automatically find the first layer in the network et get its input size.
            input_sizes={layer.name: getattr(layer, "input_size", None) for _, layer in input_layers.items()},
            output_size={layer.name: getattr(layer, "output_size", None) for _, layer in output_layers.items()},
            name=name,
            checkpoint_folder=checkpoint_folder,
            device=device,
            input_transform=input_transform,
            output_transform=output_transform,
            **kwargs
        )
        self._default_n_hidden_neurons = self.kwargs.get("n_hidden_neurons", 128)
        self.input_layers, self.hidden_layers, self.output_layers = self._layers_containers_to_modules(
            input_layers, hidden_layers, output_layers
        )
        assert len(self.get_all_layers_names()) == len(set(self.get_all_layers_names())), \
            "There are layers with the same name."

    @BaseModel.device.setter
    def device(self, device: torch.device):
        """
        Set the device of the model and all its layers.

        :param device: The device to use.
        :type device: torch.device

        :return: None
        """
        BaseModel.device.fset(self, device)
        for layer in self.get_all_layers():
            layer.device = device

    def get_all_layers(self) -> List[nn.Module]:
        """
        Get all the layers of the model as a list. The order of the layers is the same as the order of the layers in the
        model.

        :return: A list of all the layers of the model.
        :rtype: List[nn.Module]
        """
        return list(self.input_layers.values()) + list(self.hidden_layers) + list(self.output_layers.values())

    def get_layers(self, layer_names: Optional[List[str]] = None) -> List[nn.Module]:
        """
        Get the layers with the specified names.

        :param layer_names: The names of the layers to get.
        :type layer_names: Optional[List[str]]

        :return: The layers with the specified names.
        :rtype: List[nn.Module]
        """
        if layer_names is None:
            return self.get_all_layers()
        return [self.get_layer(layer_name) for layer_name in layer_names]

    def get_all_layers_names(self) -> List[str]:
        """
        Get all the names of the layers of the model. The order of the layers is the same as the order of the layers in
        the model.

        :return: A list of all the names of the layers of the model.
        :rtype: List[str]
        """
        return [layer.name for layer in self.get_all_layers()]

    def get_dict_of_layers(self) -> Dict[str, nn.Module]:
        """
        Get all the layers of the model as a dictionary. The order of the layers is the same as the order of the layers
        in the model. The keys of the dictionary are the names of the layers.

        :return: A dictionary of all the layers of the model.
        :rtype: Dict[str, nn.Module]
        """
        return {layer.name: layer for layer in self.get_all_layers()}

    def get_layer(self, name: Optional[str] = None) -> nn.Module:
        """
        Get a layer of the model. If the name is None, the first layer is returned which is useful when the model has
        only one layer.

        :param name: The name of the layer.
        :type name: str

        :return: The layer with the given name. If the name is None, the first layer is returned.
        :rtype: nn.Module
        """
        if name is None:
            return self.get_all_layers()[0]
        else:
            return self.get_dict_of_layers()[name]

    def __getitem__(self, name: Optional[str]) -> nn.Module:
        """
        Get a layer of the model. If the name is None, the first layer is returned which is useful when the model has
        only one layer.

        :param name: The name of the layer.
        :type name: str

        :return: The layer with the given name. If the name is None, the first layer is returned.
        :rtype: nn.Module
        """
        return self.get_layer(name)

    def infer_sizes_from_inputs(self, inputs: Union[Dict[str, Any], torch.Tensor]):
        """
        Infer the sizes of the inputs layers from the inputs of the network. The sizes of the inputs layers are set to
        the size of the inputs without the batch dimension.

        :param inputs: The inputs of the network.
        :type inputs: Union[Dict[str, Any], torch.Tensor]

        :return: None
        """
        if isinstance(inputs, torch.Tensor):
            inputs = {
                layer_name: inputs
                for layer_name, _ in self.input_layers.items()
            }
        self.input_sizes = {k: v.shape[1:] for k, v in inputs.items()}

    def initialize_weights_(self):
        """
        Initialize the weights of the layers of the model.

        :return: None
        """
        for layer in self.get_all_layers():
            if hasattr(layer, "initialize_weights_") and callable(layer.initialize_weights_):
                layer.initialize_weights_()

    def _format_single_inputs(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Cast the inputs to float tensor.

        :param inputs: Inputs tensor.
        :type inputs: torch.Tensor

        :return: Formatted Input tensor.
        :rtype: torch.Tensor
        """
        return inputs.float()

    def _format_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Return the formatted inputs formatted by self._format_single_inputs.

        :param inputs: Inputs dictionary.
        :type inputs: Dict[str, torch.Tensor]

        :return: Formatted inputs dictionary.
        :rtype: Dict[str, torch.Tensor]
        """
        return {k: self._format_single_inputs(in_tensor) for k, in_tensor in inputs.items()}

    def _inputs_to_dict(self, inputs: Union[Dict[str, Any], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Transform the inputs tensor into dictionary of tensors.

        :param inputs: The inputs of the network.
        :type inputs: Union[Dict[str, Any], torch.Tensor]

        :return: The transformed inputs.
        :rtype: Dict[str, torch.Tensor]
        """
        keys = list(self.input_layers.keys())
        if len(keys) == 0 and len(list(self.output_layers.keys())) > 0:
            keys = self.output_layers.keys()
        if isinstance(inputs, torch.Tensor):
            inputs = {k: inputs for k in keys}
        else:
            if set(inputs.keys()) != set(keys):
                raise ValueError("inputs must have the same keys as the input layers")
        return inputs

    def build_layers(self):
        """
        Build the layers of the model.

        :return: None
        """
        for layer in self.get_all_layers():
            if hasattr(layer, "build"):
                if callable(layer.build):
                    if not getattr(layer, "is_built", False):
                        layer.build()

    def build(self) -> 'Sequential':
        """
        Build the network and all its layers.

        :return: The network.
        :rtype: Sequential
        """
        super(Sequential, self).build()
        self._infer_and_set_sizes_of_all_layers()
        self.build_layers()
        self.initialize_weights_()
        self.device = self._device
        return self

    def _infer_and_set_sizes_of_all_layers(self):
        """
        Infer the sizes of all layers and set them.

        :return: None
        """
        inputs_layers_out_sum = 0
        inputs_sum_valid = True
        for layer_name, layer in self.input_layers.items():
            if hasattr(layer, "input_size") and layer.input_size is None:
                layer.input_size = self.input_sizes[layer_name]
            if hasattr(layer, "output_size"):
                if layer.output_size is None:
                    layer.output_size = self._default_n_hidden_neurons
                inputs_layers_out_sum += int(layer.output_size)
            else:
                inputs_sum_valid = False

        last_hidden_out_size = inputs_layers_out_sum
        for layer_idx, layer in enumerate(self.hidden_layers):
            if layer_idx == 0:
                if hasattr(layer, "input_size") and layer.input_size is None and inputs_sum_valid:
                    layer.input_size = last_hidden_out_size
            # layer.input_size = inputs_layers_out_sum
            elif (
                    hasattr(self.hidden_layers[layer_idx - 1], "output_size")
                    and hasattr(layer, "input_size")
                    and layer.input_size is None
            ):
                layer.input_size = self.hidden_layers[layer_idx - 1].output_size
            if hasattr(layer, "output_size") and layer.output_size is None:
                layer.output_size = self._default_n_hidden_neurons
            if hasattr(layer, "output_size"):
                last_hidden_out_size = int(layer.output_size)

        for layer_name, layer in self.output_layers.items():
            if hasattr(layer, "input_size") and layer.input_size is None:
                layer.input_size = last_hidden_out_size
            if hasattr(layer, "output_size") and layer.output_size is None:
                if self.output_sizes is None or self.output_sizes[layer_name] is None:
                    warnings.warn(
                        f"output_size is not set for layer {layer_name}. It will be set to {last_hidden_out_size}"
                    )
                    layer.output_size = last_hidden_out_size
                else:
                    layer.output_size = self.output_sizes[layer_name]
            if self.output_sizes is None:
                self.output_sizes = {layer_name: layer.output_size} if hasattr(layer, "output_size") else {}
            elif hasattr(layer, "output_size"):
                self.output_sizes[layer_name] = layer.output_size

    def forward(
            self,
            inputs: Union[Dict[str, Any], torch.Tensor],
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        :param inputs: The inputs to the model where the dimensions are {input_name: (batch_size, input_size)}.
        :type inputs: Union[Dict[str, Any], torch.Tensor]
        :param kwargs: Additional arguments for the forward pass.

        :return: A dictionary of outputs where the values are the names of the layers and the values are the outputs
                of the layers.
        :rtype: Dict[str, torch.Tensor]
        """
        inputs = self._inputs_to_dict(inputs)
        inputs = self.apply_input_transform(inputs)
        inputs = self._format_inputs(inputs)
        outputs: Dict[str, torch.Tensor] = {}

        features_list = []
        for layer_name, layer in self.input_layers.items():
            features = layer(inputs[layer_name])
            features_list.append(features)
        if features_list:
            forward_tensor = torch.concat(features_list, dim=-1)
        else:
            forward_tensor = torch.concat([inputs[in_name] for in_name in inputs], dim=-1)

        for layer_idx, layer in enumerate(self.hidden_layers):
            forward_tensor = layer(forward_tensor)

        for layer_name, layer in self.output_layers.items():
            out = layer(forward_tensor)
            outputs[layer_name] = out

        outputs_tensor = self.apply_output_transform(outputs)
        return outputs_tensor

    def get_raw_prediction(
            self,
            inputs: torch.Tensor,
            **kwargs
    ) -> Any:
        """
        Get the raw prediction of the model which is the output of the forward pass.

        :param inputs: inputs to the network.
        :type inputs: torch.Tensor

        :return: the raw prediction of the model.
        :rtype: Any
        """
        outputs = self(inputs.to(self._device))
        return outputs

    def get_prediction_proba(
            self,
            inputs: torch.Tensor,
            **kwargs
    ) -> Any:
        """
        Get the prediction probability of the model which is the softmax of the output of the forward pass.
        The softmax is performed on the last dimension. This method is generally used for classification.

        :param inputs: inputs to the network.
        :type inputs: torch.Tensor

        :return: the prediction probability of the model.
        :rtype: Union[Tuple[Any, Any, Any], Tuple[Any, Any], Any]
        """
        outs = self.get_raw_prediction(inputs, **kwargs)
        if isinstance(outs, (list, tuple)):
            m = outs[0]
        else:
            m = outs
        if isinstance(m, torch.Tensor):
            proba = torch.softmax(m, dim=-1)
        elif isinstance(m, dict):
            proba = {
                k: torch.softmax(v, dim=-1)
                for k, v in m.items()
            }
        else:
            raise ValueError("m must be a torch.Tensor or a dictionary")
        return proba

    def get_prediction_log_proba(
            self,
            inputs: torch.Tensor,
            **kwargs
    ) -> Union[Tuple[Tensor, Any, Any], Tuple[Tensor, Any], Tensor]:
        """
        Get the prediction log probability of the model which is the log softmax of the output of the forward pass.
        The log softmax is performed on the last dimension. This method is generally used for training in classification
        task.

        :param inputs: inputs to the network.
        :type inputs: torch.Tensor

        :return: the prediction log probability of the model.
        :rtype: Union[tuple[Tensor, Any, Any], tuple[Tensor, Any], Tensor]
        """
        outs = self.get_raw_prediction(inputs, **kwargs)
        if isinstance(outs, (list, tuple)):
            m = outs[0]
        else:
            m = outs
        if isinstance(m, torch.Tensor):
            log_proba = F.log_softmax(m, dim=-1)
        elif isinstance(m, dict):
            log_proba = {
                k: F.log_softmax(v, dim=-1)
                for k, v in m.items()
            }
        else:
            raise ValueError("m must be a torch.Tensor or a dictionary")
        return log_proba

    def get_and_reset_regularization_loss(self) -> torch.Tensor:
        """
        Get the regularization loss as a sum of all the regularization losses of the layers. Then reset the
        regularization losses.

        :return: the regularization loss.
        :rtype: torch.Tensor
        """
        warnings.warn(
            "This method is deprecated and will be removed in the next version. Use get_regularization_loss instead.",
            DeprecationWarning
        )
        regularization_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        for layer in self.get_all_layers():
            if hasattr(layer, "get_and_reset_regularization_loss") and callable(layer.get_and_reset_regularization_loss):
                regularization_loss += layer.get_and_reset_regularization_loss()
        return regularization_loss


