from typing import Optional

import numpy as np
import torch


@torch.no_grad()
def dale_(
        tensor: torch.Tensor,
        inh_ratio: float = 0.5,
        rho: float = 0.99,
        inh_first: bool = True,
        seed: Optional[int] = None
) -> torch.Tensor:
    """
    Initialize a tensor with Dale's law. As default it is half excitatory, half inhibitory. The connections in the
    tensor are considered as i (row) -> j (col).

    :param tensor: tensor to initialize. This tensor will be modified inplace.
    :type tensor: torch.Tensor
    :param inh_ratio: ratio of inhibitory connections. Must be between 0 and 1. Default is 0.5.
    :type inh_ratio: float
    :param rho: The connectivity ratio. Must be between 0 and 1. If rho = 1, the tensor will be fully connected.
        Default is 0.99.
    :type rho: float
    :param inh_first: If True, the inhibitory neurons will be in the first half of the tensor. If False, the neurons
        will be shuffled. Default is True.
    :type inh_first: bool
    :param seed: seed for the random number generator. If None, the seed is not set.
    :type seed: Optional[int]

    :return: The initialized tensor.
    :rtype: torch.Tensor
    """
    assert tensor.ndimension() == 2, "tensor must be 2 dimensional."
    assert 0 <= inh_ratio <= 1, "inh_ratio must be between 0 and 1."
    assert 0 <= rho <= 1, "rho must be between 0 and 1."
    rn_gen = torch.Generator()
    if seed is not None:
        rn_gen.manual_seed(seed)
    N, M = tensor.shape
    i, j = torch.triu_indices(N, M)
    N_0 = int((1 - rho) * len(i))  # Number of zero values
    values_upper = torch.cat(
        (torch.zeros(N_0), torch.normal(0, (1 / np.sqrt(N * rho * (1 - rho))), (len(i) - N_0,), generator=rn_gen))
    ).to(tensor.device)
    values_lower = torch.cat(
        (torch.zeros(N_0), torch.normal(0, (1 / np.sqrt(N * rho * (1 - rho))), (len(i) - N_0,), generator=rn_gen))
    ).to(tensor.device)
    values_upper_rn_indexes = torch.randperm(len(values_upper), generator=rn_gen)
    values_lower_rn_indexes = torch.randperm(len(values_lower), generator=rn_gen)
    tensor[i, j] = values_upper[values_upper_rn_indexes]
    tensor[j, i] = values_lower[values_lower_rn_indexes]
    tensor = torch.abs(tensor)
    if inh_first:
        inh_indexes = torch.arange(int(N * inh_ratio))
    else:
        inh_indexes = torch.randperm(N, generator=rn_gen)[:int(N * inh_ratio)]
    tensor[inh_indexes] *= -1
    return tensor


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    plt.imshow(dale_(torch.empty(100, 100), 0.5, 0.99), cmap="RdBu_r")
    plt.colorbar()
    plt.show()


