"""
NeuroTorch: A Python library for machine learning and neuroscience.
"""

import importlib_metadata

__author__ = "Jérémie Gince"
__email__ = "gincejeremie@gmail.com"
__copyright__ = "Copyright 2022, Jérémie Gince"
__license__ = "Apache 2.0"
__url__ = "https://github.com/NeuroTorch/NeuroTorch"
__package__ = "neurotorch"
__version__ = importlib_metadata.version(__package__)


import warnings

from . import init, utils
from .callbacks import (
    CheckpointManager,
    LoadCheckpointMode,
    TrainingHistory,
)
from .dimension import (
    Dimension,
    DimensionLike,
    DimensionProperty,
    DimensionsLike,
    Size,
)
from .learning_algorithms import (
    BPTT,
    RLS,
    TBPTT,
    Eprop,
)
from .metrics import (
    losses,
)
from .modules.layers import (
    ALIFLayer,
    ALIFLayerLPF,
    LayerType,
    LIFLayer,
    LIFLayerLPF,
    LILayer,
    Linear,
    SpyALIFLayer,
    SpyALIFLayerLPF,
    SpyLIFLayer,
    SpyLIFLayerLPF,
    SpyLILayer,
    WilsonCowanCURBDLayer,
    WilsonCowanLayer,
)
from .modules.sequential import (
    Sequential,
)
from .modules.sequential_rnn import (
    SequentialRNN,
)
from .regularization import (
    L1,
    L2,
    RegularizationList,
)
from .regularization.connectome import (
    DaleLaw,
    DaleLawL2,
)
from .trainers import (
    ClassificationTrainer,
    RegressionTrainer,
    Trainer,
    TrainingState,
)
from .transforms import (
    IdentityTransform,
    ToDevice,
    ToTensor,
    to_numpy,
    to_tensor,
)
from .utils import (
    set_seed,
)
from .visualisation import (
    Visualise,
    VisualiseKMeans,
    VisualisePCA,
    VisualiseUMAP,
)

warnings.filterwarnings("ignore", category=Warning, module="docutils")
warnings.filterwarnings("ignore", category=Warning, module="sphinx")
