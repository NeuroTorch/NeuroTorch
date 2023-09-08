import pickle
import time
import warnings
from typing import Callable, Dict, List, Any, Tuple, Union, Iterable, Optional, Sequence

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt

from ..transforms.base import to_tensor, to_numpy


def set_seed(seed: int):
    """
    Set the seed of the random number generator.

    :param seed: The seed to set.
    """
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def unitary_rn_normal_matrix(n: int, m: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    max_dim, min_dim = max(n, m), min(n, m)

    # rn_matrix = torch.randn((n, m), generator=generator)
    # u, s, v = torch.linalg.svd(rn_matrix, full_matrices=True)
    # unitary_rn_matrix = u @ torch.eye(*rn_matrix.shape) @ v

    # rn_matrix = torch.randn((n, m), generator=generator)
    # u, s, v = torch.linalg.svd(rn_matrix, full_matrices=True)
    # eye = torch.eye(min_dim, max_dim)
    # if n > m:
    # 	unitary_rn_matrix = v @ eye @ u
    # else:
    # 	unitary_rn_matrix = u @ eye @ v

    # rn_matrix = torch.randn((max_dim, max_dim), generator=generator)
    # rn_matrix[:max_dim-min_dim] = 0.0
    # u, s, v = torch.linalg.svd(rn_matrix, full_matrices=False)
    # unitary_rn_matrix = v[:min_dim]

    rn_matrix = torch.randn((n, m), generator=generator)
    u, s, v = torch.linalg.svd(rn_matrix, full_matrices=False)
    unitary_rn_matrix = u @ v

    if tuple(to_numpy(unitary_rn_matrix.shape, dtype=int)) != (n, m):
        unitary_rn_matrix = unitary_rn_matrix.T
    return unitary_rn_matrix


def format_pseudo_rn_seed(seed: Optional[int] = None) -> int:
    """
    Format the pseudo random number generator seed. If the seed is None, return a pseudo random seed
    else return the given seed.

    :param seed: The seed to format.
    :type seed: int or None

    :return: The formatted seed.
    :rtype: int
    """
    import random
    if seed is None:
        seed = int(time.time()) + random.randint(0, np.iinfo(int).max)
    assert isinstance(seed, int), "Seed must be an integer."
    return seed




