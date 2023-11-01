import numbers
from typing import Any, Optional, Callable, Union, Dict

import numpy as np
import torch


def to_tensor(x: Any, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).type(dtype)
    elif isinstance(x, torch.Tensor):
        return x.type(dtype)
    elif isinstance(x, numbers.Number):
        return torch.tensor(x, dtype=dtype)
    elif isinstance(x, dict):
        return {k: to_tensor(v, dtype=dtype) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)([to_tensor(v, dtype=dtype) for v in x])
    elif not isinstance(x, torch.Tensor):
        try:
            return torch.tensor(x, dtype=dtype)
        except Exception as e:
            raise ValueError(f"Unsupported type {type(x)}") from e
    raise ValueError(f"Unsupported type {type(x)}")


def to_numpy(x: Any, dtype=np.float32):
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=dtype)
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, numbers.Number):
        return x
    elif isinstance(x, dict):
        return {k: to_numpy(v, dtype=dtype) for k, v in x.items()}
    elif not isinstance(x, torch.Tensor):
        try:
            return np.asarray(x, dtype=dtype)
        except Exception as e:
            raise ValueError(f"Unsupported type {type(x)}") from e
    raise ValueError(f"Unsupported type {type(x)}")


class ToDevice(torch.nn.Module):
    def __init__(self, device: torch.device, non_blocking: bool = True):
        super().__init__()
        self._device = device
        self.non_blocking = non_blocking
        self.to(device)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self._device = device
        self.to(device)

    def to(self, device: Optional[Union[int, torch.device]], non_blocking: Optional[bool] = None, *args, **kwargs):
        self._device = device
        if non_blocking is None:
            non_blocking = self.non_blocking
        else:
            self.non_blocking = non_blocking
        return super().to(device=device, non_blocking=non_blocking, *args, **kwargs)

    def forward(self, x: torch.Tensor):
        if x is None:
            return x
        if not isinstance(x, torch.Tensor):
            if isinstance(x, dict):
                return {k: self.forward(v) for k, v in x.items()}
            elif isinstance(x, list):
                return [self.forward(v) for v in x]
            elif isinstance(x, tuple):
                return tuple(self.forward(v) for v in x)
            else:
                return x
        return x.to(self.device, non_blocking=self.non_blocking)

    def __repr__(self):
        return f"ToDevice({self.device}, async={self.non_blocking})"


class ToTensor(torch.nn.Module):
    def __init__(self, dtype=torch.float32, device: Optional[torch.device] = None):
        super().__init__()
        self.dtype = dtype
        self.device = device
        if self.device is None:
            self.to_device = None
        else:
            self.to_device = ToDevice(self.device)

    def forward(self, x: Any) -> torch.Tensor:
        x = to_tensor(x, self.dtype)
        if self.to_device is not None:
            x = self.to_device(x)
        return x


class IdentityTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Any):
        return x


class LinearRateToSpikes(torch.nn.Module):
    def __init__(
            self,
            n_steps: int,
            *,
            data_min: float = 0.0,
            data_max: float = 1.0,
            epsilon: float = 1e-6,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.data_min = to_tensor(data_min)
        self.data_max = to_tensor(data_max)
        self.epsilon = epsilon

    def __repr__(self):
        return f"{self.__class__.__name__}(n_steps={self.n_steps})"

    def firing_periods_to_spikes(self, firing_periods: np.ndarray) -> np.ndarray:
        firing_periods = np.floor(firing_periods).astype(int)
        ones_mask = firing_periods == 1
        firing_periods[firing_periods > self.n_steps] = self.n_steps
        firing_periods[firing_periods < 1] = 1
        time_dim = np.expand_dims(np.arange(1, self.n_steps+1), axis=tuple(np.arange(firing_periods.ndim)+1))
        spikes = (time_dim % firing_periods) == 0
        spikes[0, ones_mask] = 0
        return spikes.astype(float)

    def forward(self, x):
        x = to_tensor(x)
        device = x.device
        squeeze_flag = False
        if x.dim() < 1:
            x = x.unsqueeze(0)
            squeeze_flag = True
        self.data_min = torch.minimum(x, self.data_min)
        self.data_max = torch.maximum(x, self.data_max)
        x = (x - self.data_min) / (self.data_max - self.data_min + self.epsilon)
        periods = torch.floor((1 - x) * self.n_steps).type(torch.long)
        spikes = to_tensor(self.firing_periods_to_spikes(periods.cpu().numpy()))
        if squeeze_flag:
            spikes = spikes.squeeze()
        return spikes.to(device)


class ConstantValuesTransform(torch.nn.Module):
    def __init__(self, n_steps: int, batch_wise: bool = True):
        super().__init__()
        self.n_steps = n_steps
        self.batch_wise = batch_wise

    def __repr__(self):
        return f"{self.__class__.__name__}(n_steps={self.n_steps})"

    def forward(self, x: Any):
        x = to_tensor(x)
        if self.batch_wise:
            if x.ndim < 3:
                x = x.unsqueeze(1)
            return x.repeat(1, self.n_steps, 1)
        if x.ndim < 2:
            x = x.unsqueeze(0)
        return x.repeat(self.n_steps, 1)


class MaybeSoftmax(torch.nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        from ..utils import maybe_apply_softmax
        return maybe_apply_softmax(x, self.dim)


class ReduceMax(torch.nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        from ..utils import unpack_out_hh
        out, hh = unpack_out_hh(x)
        if isinstance(out, torch.Tensor):
            out_max, _ = torch.max(out, dim=self.dim)
        elif isinstance(out, dict):
            out_max = {
                k: torch.max(v, dim=self.dim)[0]
                for k, v in out.items()
            }
        else:
            raise ValueError("Inputs must be a torch.Tensor or a dictionary.")
        return out_max

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class ReduceMean(torch.nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        from ..utils import unpack_out_hh
        out, hh = unpack_out_hh(x)
        if isinstance(out, torch.Tensor):
            out_mean = torch.mean(out, dim=self.dim)
        elif isinstance(out, dict):
            out_mean = {
                k: torch.mean(v, dim=self.dim)
                for k, v in out.items()
            }
        else:
            raise ValueError("Inputs must be a torch.Tensor or a dictionary.")
        return out_mean

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class ReduceSum(torch.nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        from ..utils import unpack_out_hh
        out, hh = unpack_out_hh(x)
        if isinstance(out, torch.Tensor):
            out_sum = torch.sum(out, dim=self.dim)
        elif isinstance(out, dict):
            out_sum = {
                k: torch.sum(v, dim=self.dim)
                for k, v in out.items()
            }
        else:
            raise ValueError("Inputs must be a torch.Tensor or a dictionary.")
        return out_sum

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class ReduceFuncTanh(torch.nn.Module):
    """
    Applies a reduction function to the output of a recurrent layer and then applies a tanh activation.
    """
    def __init__(
            self,
            reduce_func: Callable[
                [Union[Dict[str, torch.Tensor], torch.Tensor]],
                Union[Dict[str, torch.Tensor], torch.Tensor]
            ]
    ):
        """
        Constructor of the ReduceFuncTanh class.

        :param reduce_func: The reduction function to apply to the output of the recurrent layer. Must take a
            torch.Tensor or a dictionary of shape(s) (batch_size, seq_len, hidden_size) as input and return a
            torch.Tensor or a dictionary of shape(s) (batch_size, hidden_size) as output.
        """
        super().__init__()
        self.reduce_func = reduce_func
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        out_reduced = self.reduce_func(x)
        if isinstance(out_reduced, torch.Tensor):
            out_reduced_tanh = self.tanh(out_reduced)
        elif isinstance(out_reduced, dict):
            out_reduced_tanh = {
                k: self.tanh(v)
                for k, v in out_reduced.items()
            }
        else:
            raise ValueError("Inputs must be a torch.Tensor or a dictionary.")
        return out_reduced_tanh

