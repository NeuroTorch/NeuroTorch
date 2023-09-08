import json
import logging
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, Iterable

import torch
from torch import nn
from torchvision.transforms import Compose, Lambda
import torch.nn.functional as F

from ..callbacks import CheckpointManager, LoadCheckpointMode
from ..dimension import DimensionLike, SizeTypes, Dimension, DimensionProperty
from ..transforms import to_tensor
from ..transforms.wrappers import CallableToModuleWrapper
from ..transforms.base import IdentityTransform, ToDevice, ToTensor
from ..utils import ravel_compose_transforms, list_of_callable_to_sequential


class NamedModule(torch.nn.Module):
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self._name_is_set = False
        self.name = name
        self._name_is_default = name is None

    @property
    def name(self) -> str:
        """
        Returns the name of the module. If the name is not set, it will be set to the class name.

        :return: The name of the module.
        """
        if self._name is None:
            return self.__class__.__name__
        return self._name

    @property
    def name_is_set(self) -> bool:
        """
        Returns whether the name of the module has been set.

        :return: Whether the name of the module has been set.
        """
        return self._name_is_set

    @name.setter
    def name(self, name: str):
        """
        Sets the name of the module.

        :param name: The name of the module.
        :return: None
        """
        self._name = name
        if name is not None:
            assert isinstance(name, str), "name must be a string."
            self._name_is_set = True


class SizedModule(NamedModule):

    def __init__(
            self,
            input_size: Optional[SizeTypes] = None,
            output_size: Optional[SizeTypes] = None,
            name: Optional[str] = None,
    ):
        super().__init__(name)
        self.input_size = input_size
        self.output_size = output_size

    @property
    def input_size(self) -> Optional[Dimension]:
        if not hasattr(self, "_input_size"):
            return None
        return self._input_size

    @input_size.setter
    def input_size(self, size: Optional[SizeTypes]):
        self._input_size = self._format_size(size)

    @property
    def output_size(self) -> Optional[Dimension]:
        if not hasattr(self, "_output_size"):
            return None
        return self._output_size

    @output_size.setter
    def output_size(self, size: Optional[SizeTypes]):
        self._output_size = self._format_size(size)

    def _format_size(self, size: Optional[SizeTypes], **kwargs) -> Optional[Dimension]:
        filter_time = kwargs.get("filter_time", getattr(self, "size_filter_time", False))
        # TODO: must accept multiple time dimensions
        if size is not None:
            if isinstance(size, Iterable):
                size = [Dimension.from_int_or_dimension(s) for s in size]
                if filter_time:
                    time_dim_count = len(list(filter(lambda d: d.dtype == DimensionProperty.TIME, size)))
                    assert time_dim_count <= 1, "Size must not contain more than one Time dimension."
                    size = list(filter(lambda d: d.dtype != DimensionProperty.TIME, size))
                if len(size) == 1:
                    size = size[0]
                else:
                    raise ValueError(
                        "Size must be a single dimension or a list of 2 dimensions with a Time one "
                        "if `filter_time` is True."
                    )
            assert isinstance(size, (int, Dimension)), "Size must be an int or Dimension."
            size = Dimension.from_int_or_dimension(size)
        return size


class BaseModel(NamedModule):
    """
    This class is the base class of all models.

    :Attributes:
        - input_sizes: The input sizes of the model.
        - input_transform (torch.nn.ModuleDict): The transforms to apply to the inputs.
        - output_sizes: The output size of the model.
        - output_transform (torch.nn.ModuleDict): The transforms to apply to the outputs.
        - name: The name of the model.
        - checkpoint_folder: The folder where the checkpoints are saved.
        - kwargs: Additional arguments.

    """
    @staticmethod
    def _format_sizes(sizes: Union[Dict[str, DimensionLike], SizeTypes]) -> Dict[str, int]:
        if isinstance(sizes, dict):
            return sizes
        elif isinstance(sizes, list):
            return {
                f"{i}": s
                for i, s in enumerate(sizes)
            }
        else:
            return {
                "0": sizes
            }

    def __init__(
            self,
            input_sizes: Optional[Union[Dict[str, DimensionLike], SizeTypes]] = None,
            output_size: Optional[Union[Dict[str, DimensionLike], SizeTypes]] = None,
            name: str = "BaseModel",
            checkpoint_folder: str = "checkpoints",
            device: torch.device = None,
            input_transform: Union[Dict[str, Callable], List[Callable]] = None,
            output_transform: Union[Dict[str, Callable], List[Callable]] = None,
            **kwargs
    ):
        """
        Constructor of the BaseModel class. This class is the base class of all models.

        :param input_sizes: The input sizes of the model.
        :type input_sizes: Union[Dict[str, DimensionLike], SizeTypes]
        :param output_size: The output size of the model.
        :type output_size: Union[Dict[str, DimensionLike], SizeTypes]
        :param name: The name of the model.
        :type name: str
        :param checkpoint_folder: The folder where the checkpoints are saved.
        :type checkpoint_folder: str
        :param device: The device of the model. If None, the default device is used.
        :type device: torch.device
        :param input_transform: The transforms to apply to the inputs. The input_transform must work batch-wise.
        :type input_transform: Union[Dict[str, Callable], List[Callable]]
        :param output_transform: The transforms to apply to the outputs. The output_transform must work batch-wise.
        :type output_transform: Union[Dict[str, Callable], List[Callable]]

        :keyword kwargs: Additional arguments.
        """
        super(BaseModel, self).__init__(name=name)
        self._is_built = False
        self._given_input_transform = input_transform
        self._given_output_transform = output_transform
        self.input_transform: torch.nn.ModuleDict = None
        self.output_transform: torch.nn.ModuleDict = None
        self.input_sizes = input_sizes
        self.output_sizes = output_size
        self.name = name
        self.checkpoint_folder = checkpoint_folder
        self.kwargs = kwargs
        self._device = device
        if self._device is None:
            self._set_default_device_()
        self._to_device_transform = ToDevice(self.device)

    @property
    def input_sizes(self) -> Dict[str, int]:
        return self._input_sizes

    @input_sizes.setter
    def input_sizes(self, input_sizes: Union[Dict[str, DimensionLike], SizeTypes]):
        # if self.input_sizes is not None:
        # 	raise ValueError("Input sizes can only be set once.")
        if input_sizes is not None:
            self._input_sizes = self._format_sizes(input_sizes)

    @property
    def output_sizes(self) -> Dict[str, int]:
        return self._output_sizes

    @output_sizes.setter
    def output_sizes(self, output_size: Union[Dict[str, DimensionLike], SizeTypes]):
        if output_size is not None:
            self._output_sizes = self._format_sizes(output_size)

    @property
    def _ready(self):
        is_all_not_none = all([s is not None for s in [self._input_sizes, self._output_sizes]])
        if is_all_not_none:
            is_any_none = any([s is None for s in list(self._input_sizes.values()) + list(self._output_sizes.values())])
        else:
            is_any_none = True
        return is_all_not_none and not is_any_none

    @property
    def is_built(self) -> bool:
        return self._is_built

    @property
    def checkpoints_meta_path(self) -> str:
        """
        The path to the checkpoints meta file.

        :return: The path to the checkpoints meta file.
        :rtype: str
        """
        full_filename = (
            f"{self.name}{CheckpointManager.SUFFIX_SEP}{CheckpointManager.CHECKPOINTS_META_SUFFIX}"
        )
        return f"{self.checkpoint_folder}/{full_filename}.json"

    @property
    def device(self) -> torch.device:
        """
        :return: The device of the model.
        :rtype: torch.device
        """
        return self._device

    @device.setter
    def device(self, device: torch.device):
        """
        Set the device of the network.

        :param device: The device to set.
        :type device: torch.device

        :return: None
        """
        self.to(device, non_blocking=True)

    def to(self, device: torch.device, non_blocking: bool = True, *args, **kwargs):
        self._device = device
        for module in self.modules():
            if module is not self and getattr(module, "device", device).type != device.type:
                module.to(device, non_blocking=non_blocking)
        return super(BaseModel, self).to(device=device, non_blocking=non_blocking, *args, **kwargs)

    def _make_input_transform(
            self,
            input_transform: Union[Dict[str, Callable], List[Callable]]
    ) -> torch.nn.ModuleDict:
        """
        Make the input transform containing the transforms to apply to the inputs. If the input_transform is None,
        the default transform is used. If the input_transform is a list, it is converted to a dict. If the
        input_transform is a dict, it is copied. The keys of the input_transform are the names of the inputs if the
        network has inputs else the outputs if the network has outputs.

        :param input_transform: The input transform to use.
        :type input_transform: Union[Dict[str, Callable], List[Callable]]

        :return: The input transform.
        :rtype: torch.nn.ModuleDict
        """
        if len(self.input_sizes) > 0:
            transform_keys = list(self.input_sizes.keys())
        elif len(self.output_sizes) > 0:
            transform_keys = list(self.output_sizes.keys())
        else:
            transform_keys = []

        if input_transform is None:
            input_transform = self.get_default_input_transform()
        if isinstance(input_transform, list):
            default_transform = self.get_default_input_transform()
            if len(input_transform) < len(transform_keys):
                for i in range(len(input_transform), len(transform_keys)):
                    input_transform.append(default_transform[transform_keys[i]])
            input_transform = {in_name: t for in_name, t in zip(transform_keys, input_transform)}
        elif callable(input_transform):
            input_transform = {in_name: input_transform for in_name in transform_keys}
        if isinstance(input_transform, dict):
            assert all([in_name in input_transform for in_name in transform_keys]), \
                f"Input transform must contain all input names: {transform_keys}"
        else:
            raise TypeError(f"Input transform must be a dict or a list of callables. Got {type(input_transform)}.")

        for in_name, t in input_transform.items():
            if isinstance(t, torch.nn.Module):
                input_transform[in_name] = t.to(self.device)
            else:
                input_transform[in_name] = CallableToModuleWrapper(t).to(self.device)
        return torch.nn.ModuleDict(input_transform)

    def _make_output_transform(
            self,
            output_transform: Union[Dict[str, Callable], List[Callable]]
    ) -> torch.nn.ModuleDict:
        """
        Make the output transform containing the transforms to apply to the outputs. If the output_transform is None,
        the default transform is used. If the output_transform is a list, it is converted to a dict. If the
        output_transform is a dict, it is copied. The keys of the output_transform are the names of the outputs if the
        network has outputs.

        :param output_transform: The output transform to use.
        :type output_transform: Union[Dict[str, Callable], List[Callable]]

        :return: The output transform.
        :rtype: torch.nn.ModuleDict
        """
        if len(self.output_sizes) > 0:
            transform_keys = list(self.output_sizes.keys())
        else:
            transform_keys = []

        if output_transform is None:
            output_transform = self.get_default_output_transform()
        if isinstance(output_transform, list):
            default_transform = self.get_default_output_transform()
            if len(output_transform) < len(transform_keys):
                for i in range(len(output_transform), len(transform_keys)):
                    output_transform.append(default_transform[transform_keys[i]])
            output_transform = {in_name: t for in_name, t in zip(transform_keys, output_transform)}
        elif callable(output_transform):
            output_transform = {in_name: output_transform for in_name in transform_keys}
        if isinstance(output_transform, dict):
            assert all([in_name in output_transform for in_name in transform_keys]), \
                f"Output transform must contain all output names: {transform_keys}"
        else:
            raise TypeError(f"Output transform must be a dict or a list of callables. Got {type(output_transform)}.")

        for out_name, t in output_transform.items():
            if isinstance(t, torch.nn.Module):
                output_transform[out_name] = t.to(self.device)
            else:
                output_transform[out_name] = CallableToModuleWrapper(t).to(self.device)
        return torch.nn.ModuleDict(output_transform)

    def load_checkpoint(
            self,
            checkpoints_meta_path: Optional[str] = None,
            load_checkpoint_mode: LoadCheckpointMode = LoadCheckpointMode.BEST_ITR,
            verbose: bool = True
    ) -> dict:
        """
        Load the checkpoint from the checkpoints_meta_path. If the checkpoints_meta_path is None, the default
        checkpoints_meta_path is used.

        :param checkpoints_meta_path: The path to the checkpoints meta file.
        :type checkpoints_meta_path: Optional[str]
        :param load_checkpoint_mode: The mode to use when loading the checkpoint.
        :type load_checkpoint_mode: LoadCheckpointMode
        :param verbose: Whether to print the loaded checkpoint information.
        :type verbose: bool

        :return: The loaded checkpoint information.
        :rtype: dict
        """
        if checkpoints_meta_path is None:
            checkpoints_meta_path = self.checkpoints_meta_path
        with open(checkpoints_meta_path, "r+") as jsonFile:
            info: dict = json.load(jsonFile)
        save_name = CheckpointManager.get_save_name_from_checkpoints(info, load_checkpoint_mode)
        checkpoint_path = f"{self.checkpoint_folder}/{save_name}"
        if verbose:
            logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint[CheckpointManager.CHECKPOINT_STATE_DICT_KEY], strict=True)
        return checkpoint

    def get_default_input_transform(self) -> Dict[str, nn.Module]:
        """
        Get the default input transform. The default input transform is a to tensor transform.

        :return: The default input transform.
        :rtype: Dict[str, nn.Module]
        """
        if len(self.input_sizes) > 0:
            transform_keys = list(self.input_sizes.keys())
        elif len(self.output_sizes) > 0:
            transform_keys = list(self.output_sizes.keys())
        else:
            transform_keys = []
        return {
            in_name: Compose([
                ToTensor(dtype=torch.float32),
            ])
            for in_name in transform_keys
        }

    def get_default_output_transform(self) -> Dict[str, nn.Module]:
        """
        Get the default output transform. The default output transform is an identity transform.

        :return: The default output transform.
        :rtype: Dict[str, nn.Module]
        """
        if len(self.output_sizes) > 0:
            transform_keys = list(self.output_sizes.keys())
        else:
            transform_keys = []
        return {
            in_name: IdentityTransform()
            for in_name in transform_keys
        }

    def apply_input_transform(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Apply the input transform to the inputs.

        :param inputs: dict of inputs of shape (batch_size, *input_size)
        :type inputs: Dict[str, Any]

        :return: The input of the network with the same shape as the input.
        :rtype: Dict[str, torch.Tensor]
        """
        assert all([in_name in self.input_transform for in_name in inputs]), \
            f"Inputs must be all in input names: {self.input_transform.keys()}"
        inputs = {
            in_name: self.input_transform[in_name](in_batch)
            for in_name, in_batch in inputs.items()
        }
        return inputs

    def apply_output_transform(self, outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Apply the output transform to the outputs.

        :param outputs: dict of outputs of shape (batch_size, *output_size).
        :type outputs: Dict[str, Any]

        :return: The output of the network transformed.
        :rtype: Dict[str, torch.Tensor]
        """
        assert all([out_name in self.output_transform for out_name in outputs]), \
            f"Outputs must be all in output names: {self.output_transform.keys()}"
        outputs = {
            out_name: self.output_transform[out_name](out_batch)
            for out_name, out_batch in outputs.items()
        }
        return outputs

    def _add_to_device_transform_(self):
        """
        Add the to_device_transform to the input transforms.

        :return: None
        """
        for in_name, trans in self.input_transform.items():
            list_of_transforms = ravel_compose_transforms(self.input_transform[in_name])
            list_of_transforms.append(self._to_device_transform)
            self.input_transform[in_name] = list_of_callable_to_sequential(list_of_transforms)
            trans.to(self.device)

    def _remove_to_device_transform_(self):
        """
        Remove the to_device transform from the transforms.

        :return: None
        """
        for in_name, trans in self.input_transform.items():
            if self._to_device_transform:
                list_of_transforms = ravel_compose_transforms(self.input_transform[in_name])
                list_of_transforms.remove(self._to_device_transform)
                self.input_transform[in_name] = list_of_callable_to_sequential(list_of_transforms)

    def _set_default_device_(self):
        """
        Set the default device of the network. The default device will be cuda if available and cpu otherwise.

        :return: None
        """
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def infer_sizes_from_inputs(self, inputs: Union[Dict[str, Any], torch.Tensor]):
        """
        Infer the input and output sizes from the inputs.

        :param inputs: The inputs of the network.
        :type inputs: Union[Dict[str, Any], torch.Tensor]

        :return: None
        """
        if isinstance(inputs, torch.Tensor):
            inputs = {
                "0": inputs
            }
        self.input_sizes = {k: v.shape[1:] for k, v in inputs.items()}

    def build(self, *args, **kwargs) -> 'BaseModel':
        """
        Build the network.

        :param args: Not used.
        :param kwargs: Not used.

        :return: The network.
        :rtype: BaseModel
        """
        self._is_built = True
        self.input_transform: Dict[str, Callable] = self._make_input_transform(self._given_input_transform)
        self.output_transform: Dict[str, Callable] = self._make_output_transform(self._given_output_transform)
        self._add_to_device_transform_()
        self.device = self._device
        return self

    def __call__(self, inputs: Union[Dict[str, Any], torch.Tensor], *args, **kwargs):
        if not self._is_built:
            self.infer_sizes_from_inputs(inputs)
            self.build()
        return super(BaseModel, self).__call__(inputs, *args, **kwargs)

    def forward(self, inputs: Union[Dict[str, Any], torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def get_prediction_trace(
            self,
            inputs: Union[Dict[str, Any], torch.Tensor],
            **kwargs
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get the prediction trace of the network.

        :param inputs: The inputs of the network.
        :type inputs: Union[Dict[str, Any], torch.Tensor]
        :param kwargs: Additional arguments.

        :return: The prediction trace.
        :rtype: Union[Dict[str, torch.Tensor], torch.Tensor]
        """
        raise NotImplementedError()

    def get_raw_prediction(
            self,
            inputs: torch.Tensor,
            re_outputs_trace: bool = True,
            re_hidden_states: bool = True
    ) -> Union[Tuple[Any, Any, Any], Tuple[Any, Any], Any]:
        """
        The raw prediction of the network.

        :param inputs: The inputs of the network.
        :type inputs: torch.Tensor
        :param re_outputs_trace: If True, the outputs trace will be returned.
        :type re_outputs_trace: bool
        :param re_hidden_states: If True, the hidden states will be returned.
        :type re_hidden_states: bool

        :return: The raw prediction.
        :rtype: Union[Tuple[Any, Any, Any], Tuple[Any, Any], Any]
        """
        raise NotImplementedError()

    def get_prediction_proba(
            self,
            inputs: torch.Tensor,
            re_outputs_trace: bool = True,
            re_hidden_states: bool = True
    ) -> Union[Tuple[Any, Any, Any], Tuple[Any, Any], Any]:
        """
        Get the prediction probabilities of the network.

        :param inputs: The inputs of the network.
        :type inputs: torch.Tensor
        :param re_outputs_trace: If True, the outputs trace will be returned.
        :type re_outputs_trace: bool
        :param re_hidden_states: If True, the hidden states will be returned.
        :type re_hidden_states: bool

        :return: The prediction probabilities.
        :rtype: Union[Tuple[Any, Any, Any], Tuple[Any, Any], Any]
        """
        m, *outs = self.get_raw_prediction(inputs, re_outputs_trace, re_hidden_states)
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
            return proba, *outs
        return proba

    def get_prediction_log_proba(
            self,
            inputs: torch.Tensor,
            re_outputs_trace: bool = True,
            re_hidden_states: bool = True
    ) -> Union[Tuple[Any, Any, Any], Tuple[Any, Any], Any]:
        """
        Get the prediction log probabilities of the network.

        :param inputs: The inputs of the network.
        :type inputs: torch.Tensor
        :param re_outputs_trace: If True, the outputs trace will be returned.
        :type re_outputs_trace: bool
        :param re_hidden_states: If True, the hidden states will be returned.
        :type re_hidden_states: bool

        :return: The prediction log probabilities.
        :rtype: Union[Tuple[Any, Any, Any], Tuple[Any, Any], Any]
        """
        m, *outs = self.get_raw_prediction(inputs, re_outputs_trace, re_hidden_states)
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
            return log_proba, *outs
        return log_proba

    def soft_update(self, other: 'BaseModel', tau: float = 1e-2) -> None:
        """
        Copies the weights from the other network to this network with a factor of tau.

        :param other: The other network.
        :type other: 'BaseModel'
        :param tau: The factor of the copy.
        :type tau: float

        :return: None
        """
        with torch.no_grad():
            for param, other_param in zip(self.parameters(), other.parameters()):
                param.data.copy_((1 - tau) * param.data + tau * other_param.data)

    def hard_update(self, other: 'BaseModel') -> None:
        """
        Copies the weights from the other network to this network.

        :param other: The other network.
        :type other: 'BaseModel'

        :return: None
        """
        with torch.no_grad():
            self.load_state_dict(other.state_dict())

    def to_onnx(self, in_viz=None):
        """
        Creates an ONNX model from the network.

        :param in_viz: The input to visualize.
        :type in_viz: Any

        :return: The ONNX model.
        """
        if in_viz is None:
            in_viz = torch.randn((1, self.input_sizes), device=self._device)
        torch.onnx.export(
            self,
            in_viz,
            f"{self.checkpoint_folder}/{self.name}.onnx",
            verbose=True,
            input_names=None,
            output_names=None,
            opset_version=11
        )
