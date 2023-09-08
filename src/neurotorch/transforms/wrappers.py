from typing import Callable

import torch


class CallableToModuleWrapper(torch.nn.Module):
    def __init__(self, callable_object: Callable):
        super().__init__()
        assert callable(callable_object), "The callable object must be callable."
        self.callable_object = callable_object

    def __repr__(self):
        return f"CallableToModuleWrapper({self.callable_object})"

    def forward(self, *args, **kwargs):
        return self.callable_object(*args, **kwargs)


