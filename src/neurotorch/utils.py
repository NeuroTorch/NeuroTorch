import collections.abc
import hashlib
import pickle
import time
from typing import Callable, Dict, List, Any, Tuple, Union, Iterable, Optional, Sequence

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt


def batchwise_temporal_filter(x: torch.Tensor, decay: float = 0.9):
	"""
	:param x: (batch_size, time_steps, ...)
	:param decay:
	
	:return:
	"""
	batch_size, time_steps, *_ = x.shape
	assert time_steps >= 1
	
	powers = torch.arange(time_steps, dtype=torch.float32, device=x.device).flip(0)
	weighs = torch.pow(decay, powers)
	
	x = torch.mul(x, weighs.unsqueeze(0).unsqueeze(-1))
	x = torch.sum(x, dim=1)
	return x


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
