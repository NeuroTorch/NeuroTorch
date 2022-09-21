import time
from typing import Type, Optional

import numpy as np
import torch
import neurotorch as nt
from neurotorch.regularization import BaseRegularization


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


def get_regularization(
		regularization_name: Optional[str],
		parameters,
		**kwargs
) -> Optional[BaseRegularization]:
	if regularization_name is None or not regularization_name:
		return None
	regs = regularization_name.lower().split('_')
	name_to_reg = {
		"l1": nt.L1,
		"l2": nt.L2,
		"dale": nt.DaleLaw,
	}
	return nt.RegularizationList([name_to_reg[reg](parameters, **kwargs) for reg in regs])

