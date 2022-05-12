import numpy as np
import torch



def to_tensor(x, dtype=torch.float32):
	if isinstance(x, np.ndarray):
		return torch.from_numpy(x).type(dtype)
	elif not isinstance(x, torch.Tensor):
		return torch.tensor(x, dtype=dtype)
	return x.type(dtype)

