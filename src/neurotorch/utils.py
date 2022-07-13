import collections.abc
import enum
import hashlib
import os
from collections import defaultdict
from typing import Callable, Dict, List, NamedTuple, Any, Tuple, Union
import torch

import numpy as np
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


def plot_confusion_matrix(cm, classes,):
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
	return max(init_value * decay_value ** current_itr, min_value)


def get_meta_name(params: Dict[str, Any]):
	meta_name = f""
	for k, v in params.items():
		meta_name += f"{k}-{v}_"
	return meta_name[:-1]


def hash_params(params: Dict[str, Any]):
	"""
	Hash the parameters to get a unique and persistent id.
	:param params:
	:return:
	"""
	return int(hashlib.md5(get_meta_name(params).encode('utf-8')).hexdigest(), 16)


def ravel_compose_transforms(
		transform: Union[List, Tuple, torchvision.transforms.Compose, Callable]
) -> List[Callable]:
	transforms = []
	if isinstance(transform, torchvision.transforms.Compose):
		for t in transform.transforms:
			transforms.extend(ravel_compose_transforms(t))
	elif isinstance(transform, (List, Tuple)):
		for t in transform:
			transforms.extend(ravel_compose_transforms(t))
	elif callable(transform):
		transforms.append(transform)
	else:
		raise ValueError(f"Unsupported transform type: {type(transform)}")
	return transforms

