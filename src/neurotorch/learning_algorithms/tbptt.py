import warnings
from collections import defaultdict
from typing import Optional, Sequence, Union, Dict, Callable, List, Tuple, Mapping

import numpy as np
import torch
from torch.utils.hooks import RemovableHandle

from .bptt import BPTT
from ..transforms.base import to_tensor
from ..utils import (
    list_insert_replace_at,
    zero_grad_params,
    recursive_detach,
    recursive_detach_,
    unpack_out_hh,
    format_pred_batch,
    dy_dw_local,
)


class TBPTT(BPTT):
    """
    Truncated Backpropagation Through Time (TBPTT) algorithm.
    """
    def __init__(
            self,
            *,
            params: Optional[Sequence[torch.nn.Parameter]] = None,
            layers: Optional[Union[Sequence[torch.nn.Module], torch.nn.Module]] = None,
            output_layers: Optional[Union[Sequence[torch.nn.Module], torch.nn.Module]] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            criterion: Optional[Union[Dict[str, Union[torch.nn.Module, Callable]], torch.nn.Module, Callable]] = None,
            backward_time_steps: Optional[int] = None,
            optim_time_steps: Optional[int] = None,
            **kwargs
    ):
        """
        Constructor for TBPTT class.

        :param params: The parameters to optimize. If None, the parameters of the model's trainer will be used.
        :type params: Optional[Sequence[torch.nn.Parameter]]
        :param layers: The layers to apply the TBPTT algorithm to. If None, the layers of the model's trainer will be used.
        :type layers: Optional[Union[Sequence[torch.nn.Module], torch.nn.Module]]
        :param output_layers: The layers to use as output layers. If None, the output layers of the model's trainer will be used.
        :type output_layers: Optional[Union[Sequence[torch.nn.Module], torch.nn.Module]]
        :param optimizer: The optimizer to use.
        :type optimizer: Optional[torch.optim.Optimizer]
        :param criterion: The criterion to use.
        :type criterion: Optional[Union[Dict[str, Union[torch.nn.Module, Callable]], torch.nn.Module, Callable]]
        :param backward_time_steps: The number of time steps to use for the backward pass.
        :type backward_time_steps: Optional[int]
        :param optim_time_steps: The number of time steps to use for the optimizer step.
        :type optim_time_steps: Optional[int]
        :param kwargs: Additional keyword arguments.

        :keyword float auto_backward_time_steps_ratio: The ratio of the number of time steps to use for the backward pass.
                                Defaults to 0.1.
        :keyword float auto_optim_time_steps_ratio: The ratio of the number of time steps to use for the optimizer step.
                                Defaults to 0.1.
        :keyword float alpha: The alpha value to use for the exponential moving average of the gradients.
        :keyword float grad_norm_clip_value: The value to clip the gradients norm to. This parameter is used to
                        normalize the gradients of the parameters in order to help the convergence and avoid
                        overflowing. Defaults to 1.0.
        :keyword float nan: The value to use to replace the NaN values in the gradients. Defaults to 0.0.
        :keyword float posinf: The value to use to replace the inf values in the gradients. Defaults to 1.0.
        :keyword float neginf: The value to use to replace the -inf values in the gradients. Defaults to -1.0.

        :raises AssertionError: If auto_backward_time_steps_ratio is not between 0 and 1.
        :raises AssertionError: If auto_optim_time_steps_ratio is not between 0 and 1.

        .. seealso::
            :py:meth:`neurotorch.learning_algorithms.bptt.BPTT.__init__`
        """
        super(TBPTT, self).__init__(params=params, layers=layers, optimizer=optimizer, criterion=criterion, **kwargs)
        self.output_layers = output_layers
        self._hidden_layer_names = []
        self._original_forwards = {}  # {layer_name: (layer, layer.forward)}
        self._auto_set_backward_time_steps = backward_time_steps is None
        self.backward_time_steps = backward_time_steps
        self._auto_backward_time_steps_ratio = kwargs.get("auto_backward_time_steps_ratio", 0.1)
        assert 0 <= self._auto_backward_time_steps_ratio <= 1, "auto_backward_time_steps_ratio must be between 0 and 1"
        self._auto_set_optim_time_steps = optim_time_steps is None
        self.optim_time_steps = optim_time_steps
        self._auto_optim_time_steps_ratio = kwargs.get("auto_optim_time_steps_ratio", self._auto_backward_time_steps_ratio)
        assert 0 <= self._auto_optim_time_steps_ratio <= 1, "auto_optim_time_steps_ratio must be between 0 and 1"
        self._data_n_time_steps = 0
        self._layers_buffer = defaultdict(list)
        self._forwards_decorated = False
        self._optim_counter = 0
        self._grads = []
        self.alpha = kwargs.get("alpha", 0.0)
        self.grad_norm_clip_value = to_tensor(kwargs.get("grad_norm_clip_value", torch.inf))
        self.nan = kwargs.get("nan", 0.0)
        self.posinf = kwargs.get("posinf", 1.0)
        self.neginf = kwargs.get("neginf", -1.0)
        self.forwards_hooks: List[RemovableHandle] = []
        self._use_hooks = kwargs.get("use_hooks", False)

    def initialize_output_layers(self, trainer):
        """
        Initialize the output layers of the optimizer. Try multiple ways to identify the output layers if those are not
        provided by the user.

        :Note: Must be called before :meth:`initialize_output_params`.

        :param trainer: The trainer object.
        :return: None.
        """
        if not self.output_layers:
            self.output_layers = []
            possible_attrs = ["output_layers", "output_layer"]
            for attr in possible_attrs:
                obj = getattr(trainer.model, attr, [])
                if isinstance(obj, (Sequence, torch.nn.ModuleList)):
                    obj = list(obj)
                elif isinstance(obj, (Mapping, torch.nn.ModuleDict)):
                    obj = list(obj.values())
                elif isinstance(obj, torch.nn.Module):
                    obj = [obj]
                self.output_layers += list(obj)

        if not self.output_layers:
            raise ValueError("Could not find output layers. Please provide them manually.")

    def initialize_layers(self, trainer):
        """
        Initialize the layers of the optimizer. Try multiple ways to identify the output layers if those are not
        provided by the user.

        :param trainer: The trainer object.

        :return: None
        """
        if not self.layers:
            self.layers = []
            possible_attrs = ["input_layers", "input_layer", "hidden_layers", "hidden_layer"]
            for attr in possible_attrs:
                if hasattr(trainer.model, attr):
                    obj = getattr(trainer.model, attr, [])
                    if isinstance(obj, (Sequence, torch.nn.ModuleList)):
                        obj = list(obj)
                    elif isinstance(obj, (Mapping, torch.nn.ModuleDict)):
                        obj = list(obj.values())
                    elif isinstance(obj, torch.nn.Module):
                        obj = [obj]
                    self.layers += list(obj)
        if not self.layers:
            warnings.warn("No hidden layers found. Please provide them manually if you have any.")

    def _grads_zeros_(self):
        self._grads = [torch.zeros_like(p) for p in self.params]

    def start(self, trainer, **kwargs):
        super().start(trainer)
        self.initialize_output_layers(trainer)
        self.initialize_layers(trainer)
        self._initialize_original_forwards()

    def on_batch_begin(self, trainer, **kwargs):
        super().on_batch_begin(trainer)
        self.trainer = trainer
        if trainer.model.training:
            self._data_n_time_steps = self._get_data_time_steps_from_y_batch(
                trainer.current_training_state.y_batch, trainer.current_training_state.x_batch
            )
            self._maybe_update_time_steps()
            self.optimizer.zero_grad()
            self._grads_zeros_()
            self.decorate_forwards()

    def on_batch_end(self, trainer, **kwargs):
        super().on_batch_end(trainer)
        if trainer.model.training:
            for layer_name in self._layers_buffer:
                backward_t = len(self._layers_buffer[layer_name])
                if backward_t > 0:
                    self._backward_at_t(self._data_n_time_steps - 1, backward_t, layer_name)
                    self.optimizer.step()
        self.undecorate_forwards()
        self._layers_buffer.clear()
        self.optimizer.zero_grad()
        self._grads_zeros_()

    def _get_data_time_steps_from_y_batch(
            self,
            y_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
            x_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> int:
        """
        Get the number of time steps from the y_batch if it has more than 2 dimensions, otherwise from x_batch.

        :param y_batch: The y_batch from the current training state.
        :param x_batch: The x_batch from the current training state.
        :return: The number of time steps.
        :rtype: int
        :raises ValueError: If y_batch and x_batch are not both either torch.Tensor or dict.
        """
        time_steps = None
        if isinstance(y_batch, torch.Tensor):
            if len(y_batch.shape) > 2:
                time_steps = y_batch.shape[1]
        elif isinstance(y_batch, dict):
            if all([len(y.shape) > 2 for y in y_batch.values()]):
                time_steps = max([y.shape[1] for y in y_batch.values()])
        else:
            raise ValueError(f"y_batch must be either a torch.Tensor or a dict, but got {type(y_batch)}")
        if time_steps is None:
            if isinstance(x_batch, torch.Tensor):
                time_steps = x_batch.shape[1]
            elif isinstance(x_batch, dict):
                time_steps = max([x.shape[1] for x in x_batch.values()])
            else:
                raise ValueError(f"x_batch must be either a torch.Tensor or a dict, but got {type(x_batch)}")
        return time_steps

    def _initialize_original_forwards(self):
        """
        Initialize the original forward functions of the layers (not decorated).

        :return: None
        """
        for layer in self.trainer.model.get_all_layers():
            self._original_forwards[layer.name] = (layer, layer.forward)

    def decorate_forwards(self):
        if self.trainer.model.training:
            if not self._forwards_decorated:
                self._initialize_original_forwards()
            self._hidden_layer_names.clear()

            for layer in self.layers:
                self._hidden_layer_names.append(layer.name)
                if self._use_hooks:
                    hook = layer.register_forward_hook(self._hidden_hook, with_kwargs=True)
                    self.forwards_hooks.append(hook)
                else:
                    layer.forward = self._decorate_hidden_forward(layer.forward, layer.name)

            for layer in self.output_layers:
                if self._use_hooks:
                    hook = layer.register_forward_hook(self._output_hook, with_kwargs=True)
                    self.forwards_hooks.append(hook)
                else:
                    layer.forward = self._decorate_forward(layer.forward, layer.name)
            self._forwards_decorated = True

    def undecorate_forwards(self):
        for name, (layer, original_forward) in self._original_forwards.items():
            layer.forward = original_forward
        for hook in self.forwards_hooks:
            hook.remove()
        self.forwards_hooks.clear()
        self._forwards_decorated = False

    def _maybe_update_time_steps(self):
        if self._auto_set_backward_time_steps:
            self.backward_time_steps = max(1, int(self._auto_backward_time_steps_ratio * self._data_n_time_steps))
        if self._auto_set_optim_time_steps:
            self.optim_time_steps = max(1, int(self._auto_optim_time_steps_ratio * self._data_n_time_steps))
        # if self.backward_time_steps != self.optim_time_steps:
        # 	raise NotImplementedError("backward_time_steps != optim_time_steps is not implemented yet")
        if self.backward_time_steps > self.optim_time_steps:
            raise NotImplementedError("backward_time_steps must be lower or equal to optim_time_steps.")

    def _decorate_hidden_forward(self, forward, layer_name: str) -> Callable:
        """
        Decorate the forward method of a layer to detach the hidden state.

        :param forward: The forward method to decorate.
        :param layer_name: The name of the layer.

        :return: The decorated forward method
        """
        def _forward(*args, **kwargs):
            out = forward(*args, **kwargs)
            t, forecasting = kwargs.get("t", None), kwargs.get("forecasting", False)
            if t is None:
                return out
            out_tensor, hh = unpack_out_hh(out)
            return out_tensor, recursive_detach(hh)
        return _forward

    def _decorate_forward(self, forward, layer_name: str):
        def _forward(*args, **kwargs):
            out = forward(*args, **kwargs)
            t, forecasting = kwargs.get("t", None), kwargs.get("forecasting", False)
            if t is None:
                return out
            out_tensor, hh = unpack_out_hh(out)
            list_insert_replace_at(self._layers_buffer[layer_name], t % self.backward_time_steps, out_tensor)
            self._optim_counter += 1
            if len(self._layers_buffer[layer_name]) == self.backward_time_steps:
                self._backward_at_t(t, self.backward_time_steps, layer_name)
                out = recursive_detach(out)
            if self._optim_counter >= self.optim_time_steps:
                self._make_optim_step()
            return out
        return _forward

    def _hidden_hook(self, module, args, kwargs, output) -> None:
        t, forecasting = kwargs.get("t", None), kwargs.get("forecasting", False)
        if t is None:
            return

        out_tensor, hh = unpack_out_hh(output)
        hh = recursive_detach_(hh)
        return

    def _output_hook(self, module, args, kwargs, output) -> None:
        t, forecasting = kwargs.get("t", None), kwargs.get("forecasting", False)
        if t is None:
            return

        layer_name = module.name
        out_tensor, hh = unpack_out_hh(output)
        list_insert_replace_at(self._layers_buffer[layer_name], t % self.backward_time_steps, out_tensor)
        self._optim_counter += 1
        if len(self._layers_buffer[layer_name]) == self.backward_time_steps:
            self._backward_at_t(t, self.backward_time_steps, layer_name)
            output = recursive_detach_(output)
        if self._optim_counter >= self.optim_time_steps:
            self._make_optim_step()
        return

    def _backward_at_t(self, t: int, backward_t: int, layer_name: str):
        y_batch = self._get_y_batch_slice_from_trainer((t + 1) - backward_t, t + 1, layer_name)
        pred_batch = self._get_pred_batch_from_buffer(layer_name)
        batch_loss = self.apply_criterion(pred_batch, y_batch)
        if batch_loss.grad_fn is None:
            # raise ValueError(
            #     f"batch_loss.grad_fn is None. This is probably an internal error. Please report this issue on GitHub."
            # )
            warnings.warn(
                f"batch_loss.grad_fn is None. This is probably an internal error. Please report this issue on GitHub."
            )
            self._layers_buffer[layer_name].clear()
            return

        if np.isclose(self.alpha, 0.0):
            batch_loss.backward()
        else:
            self._compute_decay_grads_(batch_loss)
            self._apply_grads()
        self._clip_grads()
        self._layers_buffer[layer_name].clear()

    def _compute_decay_grads_(self, batch_loss):
        output_grads = dy_dw_local(torch.mean(batch_loss), self.params, retain_graph=True, allow_unused=True)
        with torch.no_grad():
            self._grads = [
                self.alpha * g + torch.nan_to_num(
                    dy_dw.to(g.device),
                    nan=0.0,
                    neginf=-self.grad_norm_clip_value,
                    posinf=self.grad_norm_clip_value,
                )
                for g, dy_dw in zip(self._grads, output_grads)
            ]

    def _apply_grads(self):
        with torch.no_grad():
            for p, g in zip(self.params, self._grads):
                p.grad = g.to(p.device)

    def _clip_grads(self):
        if torch.isfinite(self.grad_norm_clip_value):
            torch.nn.utils.clip_grad_norm_(self.params, self.grad_norm_clip_value)

    def _make_optim_step(self, **kwargs):
        self.optimizer.step()
        self.optimizer.zero_grad()
        zero_grad_params(self.params)
        self._optim_counter = 0

    def _get_y_batch_slice_from_trainer(self, t_first: int, t_last: int, layer_name: str = None):
        """
        Get a slice of the y_batch from the current training state given the first and last time steps. In case
        y_batch is a dict, the slice is applied to all the values of the dict. In case y_batch is a torch.Tensor with
        less than 3 dimensions, the y_batch is stacked along the second dimension t_last - t_first times.

        :param t_first: first time step
        :param t_last: last time step
        :param layer_name: if y_batch is a dict, the slice is applied to the value corresponding to this key
        :return: a slice of the y_batch from the current training state
        """
        y_batch = self.trainer.current_training_state.y_batch.clone()
        if isinstance(y_batch, dict):
            if layer_name is None:
                y_batch = {
                    key: (
                        val[:, t_first:t_last]
                        if len(val.shape) > 2
                        else torch.stack([val for _ in range(t_last - t_first)], dim=1)
                    )
                    for key, val in y_batch.items()
                }
            else:
                y_batch = (
                    y_batch[layer_name][:, t_first:t_last]
                    if len(y_batch[layer_name].shape) > 2
                    else torch.stack([y_batch[layer_name] for _ in range(t_last - t_first)], dim=1)
                )
        else:
            y_batch = (
                y_batch[:, t_first:t_last]
                if len(y_batch.shape) > 2
                else torch.stack([y_batch for _ in range(t_last - t_first)], dim=1)
            )
        return y_batch

    def _get_pred_batch_from_buffer(self, layer_name: str):
        pred_batch = torch.stack(self._layers_buffer[layer_name], dim=1)
        return pred_batch

    def on_optimization_begin(self, trainer, **kwargs):
        y_batch = trainer.current_training_state.y_batch
        pred_batch = format_pred_batch(trainer.current_training_state.pred_batch, y_batch)
        batch_loss = self.apply_criterion(pred_batch, y_batch)
        trainer.update_state_(batch_loss=batch_loss)

    def on_optimization_end(self, trainer, **kwargs):
        super(TBPTT, self).on_optimization_end(trainer)
        self._layers_buffer.clear()

    def close(self, trainer, **kwargs):
        self.undecorate_forwards()
        super(TBPTT, self).close(trainer)

    def extra_repr(self) -> str:
        _repr = f"backward_time_steps: {self.backward_time_steps}, "
        _repr += f"optim_time_steps: {self.optim_time_steps}, "
        _repr += f"alpha: {self.alpha}, "
        _repr += f"grad_norm_clip_value: {self.grad_norm_clip_value}"
        return _repr

