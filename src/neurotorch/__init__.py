"""
NeuroTorch: A Python library for machine learning and neuroscience.
"""

__author__ = "Jérémie Gince"
__email__ = "gincejeremie@gmail.com"
__copyright__ = "Copyright 2022, Jérémie Gince"
__license__ = "Apache 2.0"
__url__ = "https://github.com/NeuroTorch/NeuroTorch"
__version__ = "0.0.1-beta5"


from .dimension import (
    Dimension,
    DimensionProperty,
    Size,
    DimensionLike,
    DimensionsLike,
)

from .modules.sequential import (
    Sequential,
)

from .modules.sequential_rnn import (
    SequentialRNN,
)

from .modules.layers import (
    LayerType,
    Linear,
    LILayer,
    LIFLayer,
    SpyLILayer,
    SpyLIFLayer,
    SpyALIFLayer,
    ALIFLayer,
    SpyLIFLayerLPF,
    SpyALIFLayerLPF,
    LIFLayerLPF,
    ALIFLayerLPF,
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
    RLS,
)

from .visualisation import (
    Visualise,
    VisualiseKMeans,
    VisualisePCA,
    VisualiseUMAP,
)

from . import utils

from .utils import (
    set_seed,
)

import warnings

warnings.filterwarnings("ignore", category=Warning, module="docutils")
warnings.filterwarnings("ignore", category=Warning, module="sphinx")
