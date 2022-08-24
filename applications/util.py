from typing import Type

import torch


def get_optimizer(optimizer_name: str) -> Type[torch.optim.Optimizer]:
	name_to_opt = {
		"sgd": torch.optim.SGD,
		"adam": torch.optim.Adam,
		"adamax": torch.optim.Adamax,
		"rmsprop": torch.optim.RMSprop,
		"adagrad": torch.optim.Adagrad,
		"adadelta": torch.optim.Adadelta,
		"adamw": torch.optim.AdamW,
	}
	return name_to_opt[optimizer_name.lower()]

