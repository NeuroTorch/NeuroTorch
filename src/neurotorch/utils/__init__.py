import enum
import warnings
from typing import Callable, List, Tuple, Union, Iterable

import torch
import torchvision

from .autograd import (
    zero_grad_params,
    dy_dw_local,
    filter_parameters,
    get_contributing_params,
    recursive_detach,
    recursive_detach_,
)
from .formatting import (
    format_pred_batch,
)
from .random import (
    set_seed,
    format_pseudo_rn_seed,
)
from .visualise import (
    legend_without_duplicate_labels_,
)
from .collections import (
    get_meta_name,
    hash_params,
    get_meta_str,
    hash_meta_str,
    save_params,
    get_all_params_combinations,
    list_of_callable_to_sequential,
    sequence_get,
    list_insert_replace_at,
    unpack_out_hh,
    unpack_tuple,
    unpack_singleton_dict,
    maybe_unpack_singleton_dict,
    mapping_update_recursively,
)
from ..transforms.base import to_tensor


class ConnectivityConvention(enum.Enum):
    """
    Enum class to specify the convention used to define the connectivity matrix.

    .. note::
        - ``ItoJ``: The connectivity matrix is defined as ``W[i, j]`` where ``i`` is the index of the pre-synaptic
                      neuron and ``j`` is the index of the post-synaptic neuron.
        - ``JtoI``: The connectivity matrix is defined as ``W[i, j]`` where ``i`` is the index of the post-synaptic
                      neuron and ``j`` is the index of the pre-synaptic neuron.

    .. note::
        The convention ``ItoJ`` is the one used by default in NeuroTorch. In this case, if the convention is not
        specified in a function, method, class or a feature, it is assumed to be ``ItoJ``.
    """
    ItoJ = 0
    JtoI = 1

    @classmethod
    def from_str(cls, convention: str):
        if convention.lower() in ["itoj", "i_to_j", "i-to-j", "i->j"]:
            return cls.ItoJ
        elif convention.lower() in ["jtoi", "j_to_i", "j-to-i", "j->i"]:
            return cls.JtoI
        else:
            raise ValueError(f"Unrecognized convention: {convention}")

    @classmethod
    def from_other(cls, other):
        if isinstance(other, str):
            return cls.from_str(other)
        elif isinstance(other, cls):
            return other
        else:
            raise ValueError(f"Unrecognized convention: {other}")

    def __str__(self):
        if self == self.ItoJ:
            return "i->j"
        elif self == self.JtoI:
            return "j->i"
        else:
            raise ValueError(f"Unrecognized convention: {self}")


def batchwise_temporal_decay(x: torch.Tensor, decay: float = 0.9):
    r"""

    Apply a decay filter to the input tensor along the temporal dimension.

    :param x: Input of shape (batch_size, time_steps, ...).
    :type x: torch.Tensor
    :param decay: Decay factor of the filter.
    :type decay: float

    :return: Filtered input of shape (batch_size, ...).
    """
    batch_size, time_steps, *_ = x.shape
    assert time_steps >= 1

    powers = torch.arange(time_steps, dtype=torch.float32, device=x.device).flip(0)
    weighs = torch.pow(decay, powers)

    x = torch.mul(x, weighs.unsqueeze(0).unsqueeze(-1))
    x = torch.sum(x, dim=1)
    return x


def batchwise_temporal_filter(x: torch.Tensor, decay: float = 0.9):
    r"""

    Apply a low-pass filter to the input tensor along the temporal dimension.

    .. math::
        \begin{equation}\label{eqn:low-pass-filter}
            \mathcal{F}_\alpha\qty(x^t) = \alpha\mathcal{F}_\alpha\qty(x^{t-1}) + x^t.
        \end{equation}
        :label: eqn:low-pass-filter

    :param x: Input of shape (batch_size, time_steps, ...).
    :type x: torch.Tensor
    :param decay: Decay factor of the filter.
    :type decay: float

    :return: Filtered input of shape (batch_size, time_steps, ...).
    """
    warnings.warn(
        "This function is supposed to compute the same result as `batchwise_temporal_recursive_filter` but it "
        "doesn't. Use `batchwise_temporal_recursive_filter` instead.",
        DeprecationWarning
    )
    batch_size, time_steps, *_ = x.shape
    assert time_steps >= 1

    # TODO: check if this is correct
    powers = torch.arange(time_steps, dtype=torch.float32, device=x.device).flip(0)
    weighs = torch.pow(decay, powers)

    y = torch.mul(x, weighs.unsqueeze(0).unsqueeze(-1))
    y = torch.cumsum(y, dim=1)
    return y


def batchwise_temporal_recursive_filter(x, decay: float = 0.9):
    r"""
    Apply a low-pass filter to the input tensor along the temporal dimension recursively.

    .. math::
        \begin{equation}\label{eqn:low-pass-filter}
            \mathcal{F}_\alpha\qty(x^t) = \alpha\mathcal{F}_\alpha\qty(x^{t-1}) + x^t.
        \end{equation}
        :label: eqn:low-pass-filter

    :param x: Input of shape (batch_size, time_steps, ...).
    :type x: torch.Tensor
    :param decay: Decay factor of the filter.
    :type decay: float

    :return: Filtered input of shape (batch_size, time_steps, ...).
    """
    y = to_tensor(x).detach().clone()
    batch_size, time_steps, *_ = x.shape
    assert time_steps >= 1
    fx = 0.0
    for t in range(time_steps):
        fx = decay * fx + y[:, t]
        y[:, t] = fx
    return y


def linear_decay(init_value, min_value, decay_value, current_itr):
    return max(init_value * decay_value**current_itr, min_value)


def ravel_compose_transforms(
        transform: Union[List, Tuple, torchvision.transforms.Compose, Callable, torch.nn.ModuleList]
) -> List[Callable]:
    transforms = []
    if isinstance(transform, torchvision.transforms.Compose):
        for t in transform.transforms:
            transforms.extend(ravel_compose_transforms(t))
    elif isinstance(transform, (List, Tuple)):
        for t in transform:
            transforms.extend(ravel_compose_transforms(t))
    elif isinstance(transform, torch.nn.Module) and isinstance(transform, Iterable):
        for t in transform:
            transforms.extend(ravel_compose_transforms(t))
    elif callable(transform):
        transforms.append(transform)
    else:
        raise ValueError(f"Unsupported transform type: {type(transform)}")
    return transforms


def maybe_apply_softmax(x, dim: int = -1):
    """
    Apply softmax to x if x is not l1 normalized.

    :Note: The input will be cast to tensor bye the transform `to_tensor`.

    :param x: The tensor to apply softmax to.
    :param dim: The dimension to apply softmax to.
    :return: The softmax applied tensor.
    """
    from ..transforms.base import to_tensor
    # from torch.distributions import constraints
    out = to_tensor(x)
    # constraint = constraints.simplex
    all_positive = torch.all(out >= 0)
    dim_sum = torch.sum(out, dim=dim)
    l1_normalized = torch.allclose(dim_sum, torch.ones_like(dim_sum), atol=1e-6)
    # if constraint.check(out):
    if all_positive and l1_normalized:
        # if torch.allclose(out, out / out.sum(dim=dim, keepdim=True), atol=1e-6):
        return out
    else:
        return torch.nn.functional.softmax(out, dim=dim)


def clip_tensors_norm_(
        tensors: Union[torch.Tensor, Iterable[torch.Tensor]],
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False
) -> torch.Tensor:
    r"""Clips norm of an iterable of tensors.

    This function is a clone from torch.nn.utils.clip_grad_norm_ with the difference that it
    works on tensors instead of parameters.

    The norm is computed over all tensors together, as if they were
    concatenated into a single vector.

    Args:
        tensors (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have data normalized
        max_norm (float or int): max norm of the data
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the data from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False

    Returns:
        Total norm of the tensors (viewed as a single vector).
    """
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(tensors) == 0:
        return torch.tensor(0.)
    device = tensors[0].device
    if norm_type == torch.inf:
        norms = [t.detach().abs().max().to(device) for t in tensors]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(t.detach(), norm_type).to(device) for t in tensors]), norm_type)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`'
        )
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for t in tensors:
        t.detach().mul_(clip_coef_clamped.to(t.device))
    return total_norm
