from typing import Optional

import torch
from .base import NamedModule, SizedModule


class NamedModuleWrapper(NamedModule):
    """
    Wrapper for a module that does not inherit from NamedModule.
    """

    def __init__(self, module: torch.nn.Module, name: Optional[str] = None):
        super().__init__(name=name)
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class SizedModuleWrapper(SizedModule):
    """
    Wrapper for a module that does not inherit from SizedModule.
    """

    def __init__(
            self,
            module: torch.nn.Module,
            input_size: Optional[int] = None,
            output_size: Optional[int] = None,
            name: Optional[str] = None
    ):
        super().__init__(input_size=input_size, output_size=output_size, name=name)
        self.module = module

        # TODO:
        # if self.input_size is None:
        # 	self.infer_input_size()

        if self.input_size is not None and self.output_size is None:
            self.infer_output_size()

    def infer_input_size(self):
        raise NotImplementedError("Input size cannot be inferred from a module.")

    def infer_output_size(self):
        if self.input_size is None:
            raise ValueError("Input size is not defined.")
        forward_tensor = torch.zeros((1, int(self.input_size))).to(self.device)
        out = self.module(forward_tensor)
        self.output_size = out[0].shape
        return self.output_size

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
