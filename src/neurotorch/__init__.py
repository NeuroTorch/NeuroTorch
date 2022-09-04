__author__ = "Jérémie Gince"
__copyright__ = "Copyright 2022, Jérémie Gince"
__url__ = "https://github.com/JeremieGince/NeuroTorch"

"""NeuroTorch: A PyTorch-based framework for deep learning in neuroscience."""


from .version import __version__

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
	LearningType,
	LayerType,
	LILayer,
	LIFLayer,
	SpyLILayer,
	SpyLIFLayer,
	ALIFLayer,
	WilsonCowanLayer,
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

