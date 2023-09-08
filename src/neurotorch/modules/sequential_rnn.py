import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from . import (
    BaseLayer,
    LayerType,
    SpikeFuncType,
    SpikeFunction,
)
from ..dimension import Dimension
from ..transforms.base import ToDevice
from ..utils import sequence_get, unpack_out_hh
from .sequential import Sequential

Acceptable_Spike_Func = Union[Type[SpikeFunction], SpikeFuncType]
Acceptable_Spike_Funcs = Union[Acceptable_Spike_Func, Iterable[Acceptable_Spike_Func]]
Acceptable_Layer_Type = Union[Type[BaseLayer], LayerType]
Acceptable_Layer_Types = Union[Acceptable_Layer_Type, Iterable[Acceptable_Layer_Type]]
IntDimension = Union[int, Dimension]


class SequentialRNN(Sequential):
    """
    The SequentialRNN is a neural network that is constructed by stacking layers.

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
    def _format_hidden_outputs_traces(
            hidden_states: Dict[str, List[Tuple[torch.Tensor, ...]]]
    ) -> Dict[str, Tuple[torch.Tensor, ...]]:
        """
        Permute the hidden states to have a dictionary of shape {layer_name: (tensor, ...)}

        trace can be a list of :
            - Tensor -> list[torch.Tensor]
            - Tuple or list of Tensor or None -> Iterable[torch.Tensor] or Iterable[None]

        If the list has those format, it will be converted to a dictionary of shape {layer_name: (tensor, ...)}
        However, if you decide to format trace differently (empty list, numpy array ...) it won't be reshape into
        a dict. The new hidden states will therefore stay the same as the hidden_state.
        Also, please note that if all the element of your list are not the same type, it will raise an error. However,
        if you use a list of iterable, it will NOT check if all the element of the iterable are the same type. This
        decision was done to reduce the computation time. Make sure all the element of your list are the same type to
        avoid error.

        :param hidden_states: Dictionary of hidden states
        :type hidden_states: Dict[str, List[Tuple[torch.Tensor, ...]]]
        :return: Dictionary of hidden states with the shape {layer_name: (tensor, ...)}
        :rtype: Dict[str, Tuple[torch.Tensor, ...]]
        """
        new_hidden_states = {}
        for layer_name, trace in hidden_states.items():
            if len(trace) == 0:
                new_hidden_states[layer_name] = trace
                continue
            trace_element_type = type(trace[0])
            if not all(isinstance(e, trace_element_type) for e in trace):
                raise ValueError("The hidden states returned by the layers must always have the same type")
            # if trace is a list of tensors :
            if issubclass(trace_element_type, torch.Tensor):
                new_hidden_states[layer_name] = torch.stack(trace, dim=1)

            # if trace is a list of iterable: :
            elif issubclass(trace_element_type, Iterable):
                internal_trace_element_type = type(trace[0][0])
                # if the iterable is a list of None:
                if issubclass(internal_trace_element_type, type(None)):
                    new_hidden_states[layer_name] = [None] * len(trace)
                # if the iterable is a list of tensors:
                elif issubclass(internal_trace_element_type, torch.Tensor):
                    new_hidden_states[layer_name] = tuple([torch.stack(e, dim=1) for e in list(zip(*trace))])
                # If the iterable has another format, it will be kept as it is
                else:
                    new_hidden_states[layer_name] = trace
            # If the list has another format, it will be kept as it is
            else:
                new_hidden_states[layer_name] = trace
        # else (if trace is a list of scalar or a list of None):
        # new_hidden_states[layer_name] = trace
        #
        return new_hidden_states

    @staticmethod
    def _remove_init_hidden_state(
            hidden_states: Dict[str, List[Tuple[torch.Tensor, ...]]]
    ) -> Dict[str, List[Tuple[torch.Tensor, ...]]]:
        """
        Remove the initial hidden state from the hidden states.

        :param hidden_states: Dictionary of hidden states
        :type hidden_states: Dict[str, List[Tuple[torch.Tensor, ...]]]

        :return: Dictionary of hidden states without the initial hidden state
        :rtype: Dict[str, List[Tuple[torch.Tensor, ...]]]
        """
        return {
            layer_name: hidden_states[layer_name][1:]
            for layer_name in hidden_states
        }

    @staticmethod
    def _pop_memory_(memory: List[Any], memory_size: int) -> List[Any]:
        """
        Pop the memory from the list if the memory size is greater than ::attr:`_memory_size`.

        :param memory: List of memory
        :type memory: List[Any]
        :param memory_size: Size of the memory
        :type memory_size: int

        :return: List of memory without the first element
        :rtype: List[Any]
        """
        return memory[max(0, len(memory) - memory_size):]

    def __init__(
            self,
            layers: Iterable[Union[Iterable[BaseLayer], BaseLayer]],
            foresight_time_steps: int = 0,
            name: str = "SequentialRNN",
            checkpoint_folder: str = "checkpoints",
            device: Optional[torch.device] = None,
            input_transform: Optional[Union[Dict[str, Callable], List[Callable]]] = None,
            output_transform: Optional[Union[Dict[str, Callable], List[Callable]]] = None,
            **kwargs
    ):
        """
        The SequentialModel is a neural network that is constructed by stacking layers.

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

        :type layers: Iterable[Union[Iterable[BaseLayer], BaseLayer]]
        :param foresight_time_steps: The number of time steps to predict in the future. When multiple inputs or outputs
            are given, the outputs of the network are given to the inputs in the same order as they were specified in
            the construction of the network. In other words, the first output is given to the first input, the second
            output is given to the second input, and so on. If there are fewer outputs than inputs, the last inputs are
            not considered as recurrent inputs, so they are not fed.
        :type foresight_time_steps: int
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

        :keyword int out_memory_size: The size of the memory buffer for the output trace. The output of each layer is
                    stored in the memory buffer. If the memory_size is not specified, the memory_size is set to
                    foresight_time_steps if specified, otherwise to is set to infinity.
                    Reduce this number to 1 if you want to use less memory and if you
                    don't need the intermediate outputs. Default is foresight_time_steps if specified, otherwise inf.
        :keyword int hh_memory_size: The size of the memory buffer for the hidden state. The hidden state of each layer
                    is stored in the memory buffer. If the memory_size is not specified, the memory_size is set to
                    foresight_time_steps if specified, otherwise to is set
                    infinity. Reduce this number to 1 if you want to use less memory and if you don't need the
                    intermediate hidden states. Default is foresight_time_steps if specified, otherwise inf.
        :keyword Optional[torch.device] memory_device: The device to use for the memory buffer. If not specified,
                    the memory_device is set to the device of the model. To use less cuda memory, you can set the
                    memory_device to cpu. However, this will slow down the computation.
        """
        super(SequentialRNN, self).__init__(
            layers=layers,
            name=name,
            checkpoint_folder=checkpoint_folder,
            device=device,
            input_transform=input_transform,
            output_transform=output_transform,
            **kwargs
        )
        self.foresight_time_steps = foresight_time_steps
        assert self.foresight_time_steps >= 0, "foresight_time_steps must be >= 0"
        # self.n_hidden_neurons = self._format_hidden_neurons_(n_hidden_neurons)
        # self.spike_func = self._format_spike_funcs_(spike_funcs)
        # self.hidden_layer_types: List[Type] = self._format_layer_types_(hidden_layer_types)
        # self.readout_layer_type = self._format_layer_type_(readout_layer_type)  # TODO: change for multiple readout layers
        # self._add_layers_()
        if self.foresight_time_steps > 0:
            default_mem_value = self.foresight_time_steps
        else:
            default_mem_value = np.inf
        self._out_memory_size: int = self.kwargs.get("out_memory_size", default_mem_value)
        self._hh_memory_size: int = self.kwargs.get("hh_memory_size", default_mem_value)
        self._memory_device_transform = ToDevice(self.kwargs.get("memory_device", self.device))
        assert self._out_memory_size is not None and self._out_memory_size > 0, \
            "The memory size must be greater than 0 and not None."
        self._outputs_to_inputs_names_map: Optional[Dict[str, str]] = None

    @property
    def out_memory_size(self) -> int:
        """
        Get the size of the output memory buffer.

        :return: The size of the output memory buffer.
        :rtype: int
        """
        return self._out_memory_size

    @out_memory_size.setter
    def out_memory_size(self, memory_size: int):
        """
        Set the size of the output memory buffer.

        :param memory_size: The size of the output memory buffer.
        :type memory_size: int

        :return: None
        """
        assert memory_size is not None and memory_size > 0, "The memory size must be greater than 0 and not None."
        self._out_memory_size = memory_size

    @property
    def hh_memory_size(self) -> int:
        """
        Get the size of the hidden state memory buffer.

        :return: The size of the hidden state memory buffer.
        :rtype: int
        """
        return self._hh_memory_size

    @hh_memory_size.setter
    def hh_memory_size(self, memory_size: int):
        """
        Set the size of the hidden state memory buffer.

        :param memory_size: The size of the hidden state memory buffer.
        :type memory_size: int

        :return: None
        """
        assert memory_size is not None and memory_size > 0, "The memory size must be greater than 0 and not None."
        self._hh_memory_size = memory_size

    def _format_single_inputs(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Check the shape of the inputs. If the shape of the inputs is (batch_size, features), a new dimension is added
        to the front of the tensor to make it (batch_size, 1, features).
        If the shape of the inputs is (batch_size, v_time_steps, features), v_time_steps must be less are equal to
        time_steps and the inputs will be padded by zeros for time steps greater than time_steps.

        :param inputs: Inputs tensor.
        :type inputs: torch.Tensor
        :param kwargs: Additional keyword arguments.

        :keyword int time_steps: Number of time steps. Must be provided.

        :return: Formatted Input tensor.
        :rtype: torch.Tensor
        """
        time_steps = int(kwargs["time_steps"])
        with torch.no_grad():
            if inputs.ndim == 2:
                inputs = torch.unsqueeze(inputs, 1)
            # inputs = inputs.repeat(1, time_steps, 1)
            assert inputs.ndim >= 3, \
                "shape of inputs must be (batch_size, time_steps, ...) or (batch_size, nb_features)"

            t_diff = time_steps - inputs.shape[1]
            assert t_diff >= 0, "inputs time steps must me less or equal to time_steps"
            if t_diff > 0:
                zero_inputs = torch.zeros(
                    (inputs.shape[0], t_diff, *inputs.shape[2:]),
                    dtype=torch.float32,
                    device=self._device
                )
                inputs = torch.cat([inputs, zero_inputs], dim=1)
        return inputs.float()

    def _format_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Set the memory size of the sequential model if not already set. The default memory size is the number of
        time steps of the inputs. Return the formatted inputs formatted by self._format_single_inputs.

        :param inputs: Inputs dictionary.
        :type inputs: Dict[str, torch.Tensor]

        :return: Formatted inputs dictionary.
        :rtype: Dict[str, torch.Tensor]
        """
        max_time_steps = max([v.shape[1] for v in inputs.values()])
        return {k: self._format_single_inputs(in_tensor, time_steps=max_time_steps) for k, in_tensor in inputs.items()}

    def _get_time_steps_from_inputs(self, inputs: Dict[str, torch.Tensor]) -> int:
        """
        Get the number of time steps from the inputs. Make sure that all inputs have the same number of time steps.

        :param inputs: The inputs of the network.
        :type inputs: Dict[str, torch.Tensor]

        :return: The number of time steps.
        :rtype: int
        """
        time_steps_entries = [in_tensor.shape[1] for in_tensor in inputs.values()]
        assert len(set(time_steps_entries)) == 1, "inputs must have the same time steps"
        return time_steps_entries[0]

    def _init_hidden_states_memory(
            self,
            h0: Optional[Dict[str, Tuple[torch.Tensor, ...]]] = None
    ) -> Dict[str, List]:
        """
        Initialize the hidden states memory of the model.

        :param h0: The initial hidden states.
        :type h0: Optional[Dict[str, Tuple[torch.Tensor, ...]]]

        :return: The hidden states memory.
        :rtype: Dict[str, List]
        """
        if h0 is None:
            h0 = {}
        elif isinstance(h0, (tuple, list)):
            h0 = {layer_name: h0 for layer_name in self.get_all_layers_names()}
        elif isinstance(h0, torch.Tensor):
            h0 = {layer_name: (h0,) for layer_name in self.get_all_layers_names()}
        return {
            layer_name: [h0.get(layer_name, None)]
            for layer_name in self.get_all_layers_names()
        }

    def build(self) -> 'SequentialRNN':
        """
        Build the network and all its layers.

        :return: The network.
        :rtype: SequentialRNN
        """
        super(SequentialRNN, self).build()
        if self.foresight_time_steps > 0:
            self._map_outputs_to_inputs()
        return self

    def _map_outputs_to_inputs(self) -> Dict[str, str]:
        """
        Map the outputs of the model to the inputs of the model for forcasting purposes.

        :return: The mapping between the outputs and the inputs.
        :rtype: Dict[str, str]
        """
        self._outputs_to_inputs_names_map = {}
        if len(self.input_layers) == 1 and len(self.output_layers) == 1:
            in_name = list(self.input_layers.keys())[0]
            out_name = list(self.output_layers.keys())[0]
            self._outputs_to_inputs_names_map[out_name] = in_name
            assert self.input_sizes[in_name] == self.output_sizes[out_name], \
                f"input ({self.input_sizes[in_name]}) and output ({self.output_sizes[out_name]}) sizes must be the " \
                f"same when foresight_time_steps > 0."
        elif len(self.input_layers) == 0 and len(self.output_layers) >= 1:
            for out_layer_name in self._ordered_outputs_names:
                self._outputs_to_inputs_names_map[out_layer_name] = out_layer_name
        else:
            self._outputs_to_inputs_names_map: Dict[str, str] = {
                out_layer_name: in_layer_name
                for in_layer_name, out_layer_name in zip(self._ordered_inputs_names, self._ordered_outputs_names)
            }
            for out_layer_name, in_layer_name in self._outputs_to_inputs_names_map.items():
                assert self.input_sizes[in_layer_name] == self.output_sizes[out_layer_name], \
                    "input and output sizes must be the same when foresight_time_steps > 0."
        return self._outputs_to_inputs_names_map

    def _inputs_forward_(
            self,
            inputs: Dict[str, torch.Tensor],
            hidden_states: Dict[str, List],
            idx: int,
            t: Optional[int] = None,
            **forward_kwargs
    ) -> torch.Tensor:
        features_list = []
        for layer_name, layer in self.input_layers.items():
            hh = sequence_get(hidden_states.get(layer.name, []), idx=-1, default=None)
            features, hh = unpack_out_hh(layer(inputs[layer_name][:, idx], hh, t=t, **forward_kwargs))
            hidden_states[layer_name].append(self._memory_device_transform(hh))
            features_list.append(features)
        if features_list:
            forward_tensor = torch.concat(features_list, dim=1)  # TODO: devrait pas etre dim=-1 ?
        else:
            forward_tensor = torch.concat([inputs[in_name][:, idx] for in_name in inputs], dim=1)
        return forward_tensor

    def _hidden_forward_(
            self,
            forward_tensor: torch.Tensor,
            hidden_states: Dict[str, List[Tuple[torch.Tensor, ...]]],
            t: int,
            **forward_kwargs
    ) -> torch.Tensor:
        for layer_idx, layer in enumerate(self.hidden_layers):
            hh = sequence_get(hidden_states.get(layer.name, []), idx=-1, default=None)
            forward_tensor, hh = unpack_out_hh(layer(forward_tensor, hh, t=t, **forward_kwargs))
            hidden_states[layer.name].append(self._memory_device_transform(hh))
        return forward_tensor

    def _outputs_forward_(
            self,
            forward_tensor: torch.Tensor,
            hidden_states: Dict[str, List[Tuple[torch.Tensor, ...]]],
            outputs_trace: Dict[str, List[torch.Tensor]],
            t: int,
            **forward_kwargs
    ):
        for layer_name, layer in self.output_layers.items():
            hh = sequence_get(hidden_states.get(layer.name, []), idx=-1, default=None)
            out, hh = unpack_out_hh(layer(forward_tensor, hh, t=t, **forward_kwargs))
            outputs_trace[layer_name].append(self._memory_device_transform(out))
            hidden_states[layer_name].append(self._memory_device_transform(hh))
        return outputs_trace

    def _integrate_inputs_(
            self,
            inputs: Dict[str, torch.Tensor],
            hidden_states: Dict[str, List[Tuple[torch.Tensor, ...]]],
            outputs_trace: Dict[str, List[torch.Tensor]],
            time_steps: int,
    ) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[Tuple[torch.Tensor, ...]]]]:
        """
        Integration of the inputs or the initial conditions.

        :param inputs: the inputs to integrate.
        :type inputs: Dict[str, torch.Tensor]
        :param hidden_states: the hidden states of the model.
        :type hidden_states: Dict[str, List]
        :param outputs_trace: the outputs trace of the model.
        :type outputs_trace: Dict[str, List[torch.Tensor]]
        :param time_steps: the number of time steps to integrate.
        :type time_steps: int

        :return: the integrated inputs and the hidden states.
        :rtype: Tuple[Dict[str, torch.Tensor], Dict[str, List]]
        """
        for t in range(time_steps):
            forward_tensor = self._inputs_forward_(inputs, hidden_states, idx=t, t=t)
            forward_tensor = self._hidden_forward_(forward_tensor, hidden_states, t=t)
            outputs_trace = self._outputs_forward_(forward_tensor, hidden_states, outputs_trace, t=t)

            outputs_trace = {
                layer_name: self._pop_memory_(trace, self._out_memory_size)
                for layer_name, trace in outputs_trace.items()
            }
            hidden_states = {
                layer_name: self._pop_memory_(trace, self._hh_memory_size)
                for layer_name, trace in hidden_states.items()
            }

        return outputs_trace, hidden_states

    def _forecast_integration_(
            self,
            hidden_states: Dict[str, List[Tuple[torch.Tensor, ...]]],
            outputs_trace: Dict[str, List[torch.Tensor]],
            inputs_time_steps: int,
            foresight_time_steps: int,
    ) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[Tuple[torch.Tensor, ...]]]]:
        """
        Foresight prediction of the initial conditions.

        :param hidden_states: the hidden states of the model.
        :type hidden_states: Dict[str, List]
        :param outputs_trace: the outputs trace of the model.
        :type outputs_trace: Dict[str, List[torch.Tensor]]
        :param foresight_time_steps: the number of time steps to forecast.
        :type foresight_time_steps: int

        :return: the forecasted outputs and the hidden states.
        :rtype: Tuple[Dict[str, List[torch.Tensor]], Dict[str, List]]
        """
        if self._outputs_to_inputs_names_map is None:
            self._map_outputs_to_inputs()

        for tau in range(foresight_time_steps-1):
            t = inputs_time_steps + tau
            foresight_inputs_tensor = {
                self._outputs_to_inputs_names_map[layer_name]: torch.unsqueeze(trace[-1], dim=1)
                for layer_name, trace in outputs_trace.items()
            }
            forecast_kwargs = dict(forecasting=True, tau=tau)
            forward_tensor = self._inputs_forward_(foresight_inputs_tensor, hidden_states, idx=-1, t=t, **forecast_kwargs)
            forward_tensor = self._hidden_forward_(forward_tensor, hidden_states, t=t, **forecast_kwargs)
            outputs_trace = self._outputs_forward_(forward_tensor, hidden_states, outputs_trace, t=t, **forecast_kwargs)

            outputs_trace = {
                layer_name: self._pop_memory_(trace, self._out_memory_size)
                for layer_name, trace in outputs_trace.items()
            }
            hidden_states = {
                layer_name: self._pop_memory_(trace, self._hh_memory_size)
                for layer_name, trace in hidden_states.items()
            }

        return outputs_trace, hidden_states

    def forward(
            self,
            inputs: Union[Dict[str, Any], torch.Tensor],
            init_hidden_states: Optional[Dict[str, Tuple[torch.Tensor, ...]]] = None,
            **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Tuple[torch.Tensor, ...]]]:
        """
        Forward pass of the model.

        When it comes to integrate a time series:
            * We integrate the initial conditions <time_step> times.
            * We predict the remaining <forward_sight_time_steps - 1> time steps from the initial conditions
            * Please note that the last output of the integration of the initial conditions is the input for
            the integration of the remaining time steps AND also the first prediction.

            ::

                Example:
                    time_series = [t_0, t_1 ... t_N] if:
                    [t_0, t_1] -> Initial conditions, then t_1 generate the first prediction (t_2) :
                    [t_2, t_3 ... t_N] -> The remaining time steps are predicted from the initial conditions.

        :param inputs: The inputs to the model where the dimensions are
                        {input_name: (batch_size, time_steps, input_size)}. If the inputs have the shape
                        (batch_size, input_size), then the time_steps is 1. All the inputs must have the same
                        time_steps otherwise the inputs with lower time_steps will be padded with zeros.
        :type inputs: Union[Dict[str, Any], torch.Tensor]
        :param init_hidden_states: The initial hidden states of the model. The dimensions are
                        {layer_name: (h_0[batch_size, hidden_size_0], ..., h_K[batch_size, hidden_size_K])}
                        where K is the number of hidden states of the layer. If the initial hidden states are not
                        specified, the initial hidden states are set to None and will be initialized by the layer.
        :type init_hidden_states: Optional[Dict[str, Tuple[torch.Tensor, ...]]]
        :param kwargs: Additional arguments for the forward pass.

        :keyword int foresight_time_steps: The number of time steps to forecast. Default: The value of the
            attribute ::attr:`foresight_time_steps`.

        :return: A tuple of two dictionaries. The first dictionary contains the outputs of the model and the second
                        dictionary contains the hidden states of the model. The keys of the dictionaries are the
                        names of the layers. The values of the dictionaries are lists of tensors. The length of the
                        lists is the number of time steps.
        :rtype: Tuple[Dict[str, torch.Tensor], Dict[str, Tuple[torch.Tensor, ...]]]
        """
        foresight_time_steps = kwargs.get('foresight_time_steps', None)
        if foresight_time_steps is None:
            foresight_time_steps = self.foresight_time_steps

        inputs = self._inputs_to_dict(inputs)
        inputs = self.apply_input_transform(inputs)
        inputs = self._format_inputs(inputs)
        time_steps = self._get_time_steps_from_inputs(inputs)
        hidden_states = self._init_hidden_states_memory(init_hidden_states)
        outputs_trace: Dict[str, List[torch.Tensor]] = defaultdict(list)

        # TODO: Fix the time idx that is passed to the forward functions. _integrate_inputs_ and _forecast_integration_
        #   start at 0 that causes the time steps to go back to 0 (for the layers) when forecasting.

        # integration of the inputs or the initial conditions
        outputs_trace, hidden_states = self._integrate_inputs_(inputs, hidden_states, outputs_trace, time_steps)
        if self._hh_memory_size > time_steps:  # if the initial hidden state still in memory, remove it.
            hidden_states = self._remove_init_hidden_state(hidden_states)
        if foresight_time_steps > 0:
            # Foresight prediction of the initial conditions
            outputs_trace, hidden_states = self._forecast_integration_(
                hidden_states, outputs_trace, time_steps, foresight_time_steps
            )

        hidden_states = self._format_hidden_outputs_traces(hidden_states)
        outputs_trace_tensor = self.apply_output_transform({
            layer_name: torch.stack(trace, dim=1)
            for layer_name, trace in outputs_trace.items()
        })
        return outputs_trace_tensor, hidden_states

    def get_prediction_trace(
            self,
            inputs: Union[Dict[str, Any], torch.Tensor],
            **kwargs
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Returns the prediction trace for the given inputs. Method used for time series prediction.

        :param inputs: inputs to the network.
        :type inputs: Union[Dict[str, Any], torch.Tensor]
        :param kwargs: kwargs to be passed to the forward method.

        :keyword int foresight_time_steps: number of time steps to predict. Default is self.foresight_time_steps.

            :: Note: If the value of foresight_time_steps is specified, make sure that the values of the attributes
                :attr:`out_memory_size` and :attr:`hh_memory_size` are correctly set.

        :keyword bool return_hidden_states: if True, returns the hidden states of the model. Default is False.
        :keyword int trunc_time_steps: number of time steps to truncate the prediction trace. Default is None.

        :return: the prediction trace.
        :rtype: Union[Dict[str, torch.Tensor], torch.Tensor, Tuple[torch.Tensor, ...]]
        """
        foresight_time_steps = kwargs.get('foresight_time_steps', None)
        if foresight_time_steps is None:
            foresight_time_steps = self.foresight_time_steps
        trunc_time_steps = kwargs.get('trunc_time_steps', None)
        outputs_trace, hidden_states = self(inputs.to(self.device), **kwargs)
        if isinstance(outputs_trace, dict):
            if trunc_time_steps is not None:
                outputs_trace = {
                    layer_name: trace[:, -trunc_time_steps:]
                    for layer_name, trace in outputs_trace.items()
                }
            if len(outputs_trace) == 1:
                outputs_trace = outputs_trace[list(outputs_trace.keys())[0]]
        elif trunc_time_steps is not None:
            outputs_trace = outputs_trace[:, -trunc_time_steps:]
        if kwargs.get('return_hidden_states', False):
            if isinstance(hidden_states, dict):
                if trunc_time_steps is not None:
                    hidden_states = {
                        layer_name: tuple(trace_item[:, -trunc_time_steps:] for trace_item in trace)
                        for layer_name, trace in hidden_states.items()
                    }
                if len(hidden_states) == 1:
                    hidden_states = hidden_states[list(hidden_states.keys())[0]]
            elif trunc_time_steps is not None:
                hidden_states = hidden_states[:, -trunc_time_steps:]
            return outputs_trace, hidden_states
        return outputs_trace

    def get_raw_prediction(
            self,
            inputs: torch.Tensor,
            re_outputs_trace: bool = True,
            re_hidden_states: bool = True
    ) -> Union[Tuple[Any, Any], Any]:
        """
        Get the raw prediction of the model which is the output of the forward pass.

        :param inputs: inputs to the network.
        :type inputs: torch.Tensor
        :param re_outputs_trace: Whether to return the outputs trace. Default is True.
        :type re_outputs_trace: bool
        :param re_hidden_states: Whether to return the hidden states. Default is True.
        :type re_hidden_states: bool

        :return: the raw prediction of the model.
        :rtype: Union[Tuple[Any, Any], Any]
        """
        outputs_trace, hidden_states = self(inputs.to(self.device))
        if re_outputs_trace and re_hidden_states:
            return outputs_trace, hidden_states
        elif re_outputs_trace:
            return outputs_trace
        elif re_hidden_states:
            return hidden_states
        else:
            return None

    def get_fmt_prediction(
            self,
            inputs: torch.Tensor,
            lambda_func: Callable[[torch.Tensor], torch.Tensor] = lambda x: x[:, -1],
            re_outputs_trace: bool = True,
            re_hidden_states: bool = True
    ) -> Union[Tuple[Any, Any, Any], Tuple[Any, Any], Any]:
        """
        Get the prediction of the model which is the output of the forward pass and apply the max operation on the
        time dimension.

        :param inputs: inputs to the network.
        :type inputs: torch.Tensor
        :param lambda_func: the function to apply on the output trace. Default is get the last item on the time.
        :param re_outputs_trace: Whether to return the outputs trace. Default is True.
        :type re_outputs_trace: bool
        :param re_hidden_states: Whether to return the hidden states. Default is True.
        :type re_hidden_states: bool

        :return: the max prediction of the model.
        :rtype: Union[Tuple[Any, Any, Any], Tuple[Any, Any], Any]
        """

        outputs_trace, hidden_states = self(inputs.to(self.device))
        if isinstance(outputs_trace, torch.Tensor):
            item = lambda_func(outputs_trace)
        elif isinstance(outputs_trace, dict):
            item = {
                k: lambda_func(v)
                for k, v in outputs_trace.items()
            }
        else:
            raise ValueError("outputs_trace must be a torch.Tensor or a dictionary")
        if re_outputs_trace and re_hidden_states:
            return item, outputs_trace, hidden_states
        elif re_outputs_trace:
            return item, outputs_trace
        elif re_hidden_states:
            return item, hidden_states
        else:
            return item

    def get_last_prediction(
            self,
            inputs: torch.Tensor,
            re_outputs_trace: bool = True,
            re_hidden_states: bool = True
    ) -> Union[Tuple[Any, Any, Any], Tuple[Any, Any], Any]:
        """
        Get the prediction of the model which is the output of the forward pass and get the last item on the
        time dimension.

        :param inputs: inputs to the network.
        :type inputs: torch.Tensor
        :param re_outputs_trace: Whether to return the outputs trace. Default is True.
        :type re_outputs_trace: bool
        :param re_hidden_states: Whether to return the hidden states. Default is True.
        :type re_hidden_states: bool

        :return: the last prediction of the model.
        :rtype: Union[Tuple[Any, Any, Any], Tuple[Any, Any], Any]
        """

        return self.get_fmt_prediction(
            inputs,
            lambda_func=lambda x: x[:, -1],
            re_outputs_trace=re_outputs_trace,
            re_hidden_states=re_hidden_states
        )

    def get_max_prediction(
            self,
            inputs: torch.Tensor,
            re_outputs_trace: bool = True,
            re_hidden_states: bool = True
    ) -> Union[Tuple[Any, Any, Any], Tuple[Any, Any], Any]:
        """
        Get the prediction of the model which is the output of the forward pass and apply the max operation on the
        time dimension.

        :param inputs: inputs to the network.
        :type inputs: torch.Tensor
        :param re_outputs_trace: Whether to return the outputs trace. Default is True.
        :type re_outputs_trace: bool
        :param re_hidden_states: Whether to return the hidden states. Default is True.
        :type re_hidden_states: bool

        :return: the max prediction of the model.
        :rtype: Union[Tuple[Any, Any, Any], Tuple[Any, Any], Any]
        """
        return self.get_fmt_prediction(
            inputs,
            lambda_func=lambda x: torch.max(x, dim=1)[0],
            re_outputs_trace=re_outputs_trace,
            re_hidden_states=re_hidden_states
        )

    def get_mean_prediction(
            self,
            inputs: torch.Tensor,
            re_outputs_trace: bool = True,
            re_hidden_states: bool = True
    ) -> Union[Tuple[Any, Any, Any], Tuple[Any, Any], Any]:
        """
        Get the prediction of the model which is the output of the forward pass and apply the mean operation on the
        time dimension.

        :param inputs: inputs to the network.
        :type inputs: torch.Tensor
        :param re_outputs_trace: Whether to return the outputs trace. Default is True.
        :type re_outputs_trace: bool
        :param re_hidden_states: Whether to return the hidden states. Default is True.
        :type re_hidden_states: bool

        :return: the mean prediction of the model.
        :rtype: Union[Tuple[Any, Any, Any], Tuple[Any, Any], Any]
        """
        return self.get_fmt_prediction(
            inputs,
            lambda_func=lambda x: torch.mean(x, dim=1),
            re_outputs_trace=re_outputs_trace,
            re_hidden_states=re_hidden_states
        )

    def get_prediction_proba(
            self,
            inputs: torch.Tensor,
            re_outputs_trace: bool = True,
            re_hidden_states: bool = True
    ) -> Union[Tuple[Any, Any, Any], Tuple[Any, Any], Any]:
        """
        Get the prediction probability of the model which is the softmax of the output of the forward pass.
        The softmax is performed on the time dimension. This method is generally used for classification.

        :param inputs: inputs to the network.
        :type inputs: torch.Tensor
        :param re_outputs_trace: Whether to return the outputs trace. Default is True.
        :type re_outputs_trace: bool
        :param re_hidden_states: Whether to return the hidden states. Default is True.
        :type re_hidden_states: bool

        :return: the prediction probability of the model.
        :rtype: Union[Tuple[Any, Any, Any], Tuple[Any, Any], Any]
        """
        outs = self.get_max_prediction(inputs, re_outputs_trace, re_hidden_states)
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
        if re_outputs_trace or re_hidden_states:
            return proba, *outs[1:]
        return proba

    def get_prediction_log_proba(
            self,
            inputs: torch.Tensor,
            re_outputs_trace: bool = True,
            re_hidden_states: bool = True
    ) -> Union[Tuple[Tensor, Any, Any], Tuple[Tensor, Any], Tensor]:
        """
        Get the prediction log probability of the model which is the log softmax of the output of the forward pass.
        The log softmax is performed on the time dimension. This method is generally used for training in classification
        task.

        :param inputs: inputs to the network.
        :type inputs: torch.Tensor
        :param re_outputs_trace: Whether to return the outputs trace. Default is True.
        :type re_outputs_trace: bool
        :param re_hidden_states: Whether to return the hidden states. Default is True.
        :type re_hidden_states: bool

        :return: the prediction log probability of the model.
        :rtype: Union[tuple[Tensor, Any, Any], tuple[Tensor, Any], Tensor]
        """
        outs = self.get_max_prediction(inputs, re_outputs_trace, re_hidden_states)
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
        if re_outputs_trace or re_hidden_states:
            return log_proba, *outs[1:]
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


