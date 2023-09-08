from typing import List, Optional, Sequence

import torch

from ..callbacks.base_callback import BaseCallback


class LearningAlgorithm(BaseCallback):
    DEFAULT_PRIORITY = BaseCallback.DEFAULT_MEDIUM_PRIORITY

    def __init__(
            self,
            *,
            params: Optional[Sequence[torch.nn.Parameter]] = None,
            **kwargs
    ):
        """
        Constructor for LearningAlgorithm class.

        :param params: The parameters to optimize. If None, the parameters of the model's trainer will be used.
        :type params: Optional[Sequence[torch.nn.Parameter]]
        """
        super().__init__(**kwargs)
        self.kwargs = kwargs
        if params is None:
            params = []
        else:
            params = list(params)
        self.params: List[torch.nn.Parameter] = params



