"""
NeuroTorch: A PyTorch-based framework for deep learning in neuroscience.
"""

__author__ = "Jérémie Gince"
__email__ = "gincejeremie@gmail.com"
__copyright__ = "Copyright 2022, Jérémie Gince"
__license__ = "Apache 2.0"
__url__ = "https://github.com/NeuroTorch/NeuroTorch"
__version__ = "v0.0.1-beta"


from .dimension import (
	Dimension,
	DimensionProperty,
	Size,
	DimensionLike,
	DimensionsLike,
)

from .modules.sequential import (
	SequentialModel,
)

from .modules.layers import (
	LayerType,
	LILayer,
	LIFLayer,
	SpyLILayer,
	SpyLIFLayer,
	SpyALIFLayer,
	ALIFLayer,
	WilsonCowanLayer,
	WilsonCowanCURBDLayer,
)

from .regularization import (
	RegularizationList,
	L1,
	L2,
)

from .regularization.connectome import (
	DaleLaw,
	DaleLawL2,
)

from .trainers import (
	Trainer,
	ClassificationTrainer,
	RegressionTrainer,
	TrainingState,
)

from .transforms import (
	to_tensor,
	to_numpy,
	IdentityTransform,
	ToDevice,
	ToTensor,
)


from .metrics import (
	losses,
)

from .callbacks import (
	TrainingHistory,
	LoadCheckpointMode,
	CheckpointManager,
)

from . import init

from .learning_algorithms import (
	BPTT,
	TBPTT,
	Eprop,
	WeakRLS,
)

from .utils import (
	set_seed,
)

import warnings

warnings.filterwarnings("ignore", category=Warning, module="docutils")
warnings.filterwarnings("ignore", category=Warning, module="sphinx")
