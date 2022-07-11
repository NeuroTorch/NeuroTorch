import os
import pprint
from typing import Any, Dict
from collections import OrderedDict

import numpy as np
import psutil
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from dataset import WilsonCowanTimeSeries, WilsonCowanDataset, get_dataloaders
from neurotorch import Dimension, DimensionProperty
from neurotorch.callbacks import CheckpointManager, LoadCheckpointMode
from neurotorch.metrics import RegressionMetrics
from neurotorch.modules import SequentialModel
from neurotorch.modules.layers import WilsonCowanLayer
from neurotorch.trainers import RegressionTrainer
from neurotorch.utils import hash_params

i = 2  # Num of neurons
num_step = 1000
dt = 0.1
t_0 = np.random.rand(i, )
forward_weights = 8 * np.random.randn(i, i)
mu = 0
r = np.random.rand(i, ) * 2
tau = 1
time_series = WilsonCowanTimeSeries(num_step, dt, t_0, forward_weights, mu, r, tau).compute_timeseries()


def train_with_params(params: Dict[str, Any], n_iterations: int = 100, data_folder="tr_results", verbose=True):
    checkpoints_name = str(hash_params(params))
    checkpoint_folder = f"{data_folder}/{checkpoints_name}"
    os.makedirs(checkpoint_folder, exist_ok=True)
    print(f"Checkpoint folder: {checkpoint_folder}")
    dataloaders = get_dataloaders(
        time_series=params["time_series"],
        batch_size=params["batch_size"],
        train_val_split_ratio=params["train_val_split_ratio"],
        chunk_size=params["chunk_size"],
        ratio = params["ratio"],
        nb_workers=psutil.cpu_count(logical=False)
    )
    hidden_layer = [
        WilsonCowanLayer(
            input_size=params["hidden_layer_size"],
            output_size=params["hidden_layer_size"],
        )
        for _ in range(params["num_hidden_layers"])
    ]
    network = SequentialModel(
        layers=hidden_layer
    )
    network.build()
    checkpoint_manager = CheckpointManager(checkpoint_folder)
    trainer = RegressionTrainer(
        model=network,
        callbacks=checkpoint_manager
    )
    trainer.train(
        dataloaders["train"],
        dataloaders["val"],
        n_iterations=n_iterations,
        load_checkpoint_mode=LoadCheckpointMode.LAST_ITR,
        verbose=verbose
    )
    try:
        network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.BEST_ITR, verbose=verbose)
    except FileNotFoundError:
        network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.LAST_ITR, verbose=verbose)
    #show_prediction(dataloaders["test"], network)  # TODO: not implemented yet
    return OrderedDict(dict(
        network=network,
        checkpoints_name=checkpoints_name,
        mae={
            k: RegressionMetrics.mean_absolute_error(network, dataloaders[k], verbose=True, desc=f"{k}_mae")
            for k in dataloaders
        },
        mse={
            k: RegressionMetrics.mean_squared_error(network, dataloaders[k], verbose=True, desc=f"{k}_mse")
            for k in dataloaders
        },
        r2={
            k: RegressionMetrics.r2(network, dataloaders[k], verbose=True, desc=f"{k}_r2")
            for k in dataloaders
        },
        d2={
            k: RegressionMetrics.d2_tweedie(network, dataloaders[k], verbose=True, desc=f"{k}_d2")
            for k in dataloaders
        },
    ))


if __name__ == '__main__':
    results = train_with_params(
        {
            "time_series": time_series,
            "batch_size": 64,
            "train_val_split_ratio": 0.8,
            "chunk_size": 100,
            "ratio": 0.5,
            "num_hidden_layers": 1,
            "hidden_layer_size": 2
        },
        n_iterations=5,
        verbose=True
    )
    pprint.pprint(results, indent=4)





