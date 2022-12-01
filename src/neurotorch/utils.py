import collections.abc
import hashlib
import pickle
import time
import warnings
from typing import Callable, Dict, List, Any, Tuple, Union, Iterable, Optional, Sequence

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from unstable import unstable

from .transforms.base import to_tensor


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


@unstable
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


def mapping_update_recursively(d, u):
	"""
	from https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
	
	:param d: mapping item that wil be updated
	:param u: mapping item updater
	
	:return: updated mapping recursively
	"""
	for k, v in u.items():
		if isinstance(v, collections.abc.Mapping):
			d[k] = mapping_update_recursively(d.get(k, {}), v)
		else:
			d[k] = v
	return d


def plot_confusion_matrix(cm, classes, ):
	import matplotlib.pyplot as plt
	import itertools
	
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title('Confusion matrix')
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	
	fmt = 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(
			j, i,
			format(cm[i, j], fmt),
			horizontalalignment="center",
			color="white" if cm[i, j] > thresh else "black"
		)
	
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()
	plt.show()


def legend_without_duplicate_labels_(ax: plt.Axes):
	handles, labels = ax.get_legend_handles_labels()
	unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
	ax.legend(*zip(*unique))


def linear_decay(init_value, min_value, decay_value, current_itr):
	return max(init_value * decay_value**current_itr, min_value)


def get_meta_name(params: Dict[str, Any]):
	meta_name = f""
	keys = sorted(list(params.keys()))
	for k in keys:
		meta_name += f"{k}-{params[k]}_"
	return meta_name[:-1]


def hash_params(params: Dict[str, Any]):
	"""
	Hash the parameters to get a unique and persistent id.
	
	:param params: The parameters to hash.
	
	:return: The hash of the parameters.
	"""
	return int(hashlib.md5(get_meta_name(params).encode('utf-8')).hexdigest(), 16)


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


def save_params(params: Dict[str, Any], save_path: str):
	"""
	Save the parameters in a file.
	
	:param save_path: The path to save the parameters.
	:param params: The parameters to save.
	
	:return: The path to the saved parameters.
	"""
	pickle.dump(params, open(save_path, "wb"))
	return save_path


def get_all_params_combinations(params_space: Dict[str, Any]) -> List[Dict[str, Any]]:
	"""
	Get all possible combinations of parameters.
	
	:param params_space: Dictionary of parameters.
	
	:return: List of dictionaries of parameters.
	"""
	import itertools
	# get all the combinaison of the parameters
	all_params = list(params_space.keys())
	all_params_values = list(params_space.values())
	all_params_combinaison = list(map(lambda x: list(x), list(itertools.product(*all_params_values))))
	
	# create a list of dict of all the combinaison
	all_params_combinaison_dict = list(map(lambda x: dict(zip(all_params, x)), all_params_combinaison))
	return all_params_combinaison_dict


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


def list_of_callable_to_sequential(callable_list: List[Callable]) -> torch.nn.Sequential:
	"""
	Convert a list of callable to a list of modules.
	
	:param callable_list: List of callable.
	
	:return: List of modules.
	"""
	from neurotorch.transforms.wrappers import CallableToModuleWrapper
	return torch.nn.Sequential(*[
			c if isinstance(c, torch.nn.Module) else CallableToModuleWrapper(c)
			for c in callable_list
		])


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


def sequence_get(__sequence: Sequence, idx: int, default: Any = None) -> Any:
	try:
		return __sequence[idx]
	except IndexError:
		return default


def list_insert_replace_at(__list: List, idx: int, value: Any):
	"""
	Insert a value at a specific index. If there is already a value at this index, replace it.
	
	:param __list: The list to modify.
	:param idx: The index to insert the value.
	:param value: The value to insert.
	"""
	if idx < len(__list):
		__list[idx] = value
	else:
		__list.extend([None] * (idx - len(__list)))
		__list.append(value)


def zero_grad_params(params: Iterable[torch.nn.Parameter]):
	"""
	Set the gradient of the parameters to zero.
	
	:param params: The parameters to set the gradient to zero.
	"""
	for p in params:
		if p.grad is not None:
			p.grad.detach_()
			p.grad.zero_()


def compute_jacobian(
		*,
		model: Optional[torch.nn.Module] = None,
		params: Optional[Iterable[torch.nn.Parameter]] = None,
		x: Optional[torch.Tensor] = None,
		y: Optional[torch.Tensor] = None,
		strategy: str = "slow",
):
	"""
	Compute the jacobian of the model with respect to the parameters.
	
	# TODO: check https://medium.com/@monadsblog/pytorch-backward-function-e5e2b7e60140
	# TODO: see https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py
	
	:param model: The model to compute the jacobian.
	:param params: The parameters to compute the jacobian with respect to. If None, compute the jacobian
		with respect to all the parameters of the model.
	:param x: The input to compute the jacobian. If None, use y instead.
	:param y: The output to compute the jacobian. If None, use x instead.
	:param strategy: The strategy to use to compute the jacobian. Can be "slow" or "fast". At this time the only
		strategy implemented is "slow".
	
	:return: The jacobian.
	"""
	if params is None:
		assert model is not None, "If params is None, model must be provided."
		params = model.parameters()
	zero_grad_params(params)
	
	if y is not None:
		if strategy.lower() == "fast":
			y.backward(torch.ones_like(y))
			jacobian = [p.grad.view(-1) for p in params]
		elif strategy.lower() == "slow":
			jacobian = [[] for _ in range(len(list(params)))]
			grad_outputs = torch.eye(y.shape[-1], device=y.device)
			for i in range(y.shape[-1]):
				zero_grad_params(params)
				y.backward(grad_outputs[i], retain_graph=True)
				for p_idx, param in enumerate(params):
					jacobian[p_idx].append(param.grad.view(-1).detach().clone())
			jacobian = [torch.stack(jacobian[i], dim=-1).T for i in range(len(list(params)))]
		else:
			raise ValueError(f"Unsupported strategy: {strategy}")
	elif x is not None:
		jacobian = torch.autograd.functional.jacobian(model, x, params)
	else:
		raise ValueError("Either x or y must be provided.")
	return jacobian


def vmap(f):
	# TODO: replace by torch.vmap when it is available
	def wrapper(batch_tensor):
		return torch.stack([f(batch_tensor[i]) for i in range(batch_tensor.shape[0])])
	return wrapper


def unpack_out_hh(out):
	"""
	Unpack the output of a recurrent network.
	
	:param out: The output of a recurrent network.
	
	:return: The output of the recurrent network with the hidden state.
				If there is no hidden state, consider it as None.
	"""
	out_tensor, hh = None, None
	if isinstance(out, (tuple, list)):
		if len(out) == 2:
			out_tensor, hh = out
		elif len(out) == 1:
			out_tensor = out[0]
		elif len(out) > 2:
			out_tensor, *hh = out
	else:
		out_tensor = out
	
	return out_tensor, hh


def filter_parameters(
		parameters: Union[Sequence[torch.nn.Parameter], torch.nn.ParameterList],
		requires_grad: bool = True
) -> List[torch.nn.Parameter]:
	"""
	Filter the parameters by their requires_grad attribute.
	
	:param parameters: The parameters to filter.
	:param requires_grad: The value of the requires_grad attribute to filter.
	
	:return: The filtered parameters.
	"""
	return [p for p in parameters if p.requires_grad == requires_grad]