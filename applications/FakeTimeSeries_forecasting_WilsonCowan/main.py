import os
import pprint
from typing import Any, Dict
from collections import OrderedDict

import numpy as np
import psutil
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from applications.sinus_spikes.dataset import SinusSpikesDataset, get_dataloaders
from neurotorch import Dimension, DimensionProperty
from neurotorch.callbacks import CheckpointManager, LoadCheckpointMode
from neurotorch.metrics import RegressionMetrics
from neurotorch.modules import SequentialModel
from neurotorch.modules.layers import WilsonCowanLayer
from neurotorch.trainers import RegressionTrainer
from neurotorch.utils import hash_params

