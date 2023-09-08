import json
import pprint
from collections import OrderedDict

import torch
from pythonbasictools.device import log_device_setup, DeepLib
from pythonbasictools.logging import logs_file_setup

from dataset import get_dataloaders
import neurotorch as nt
from neurotorch import Dimension, DimensionProperty
from neurotorch.modules import HeavisideSigmoidApprox
from neurotorch.callbacks import CheckpointManager, LoadCheckpointMode
from neurotorch.metrics import ClassificationMetrics
from neurotorch.modules import SequentialRNN
from neurotorch.modules.layers import SpyLILayer
from neurotorch.trainers import ClassificationTrainer


if __name__ == '__main__':
    logs_file_setup("heidelberg", add_stdout=False)
    log_device_setup(deepLib=DeepLib.Pytorch)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)

    nt.set_seed(0)

    # Training parameters
    n_iterations = 200
    batch_size = 256
    learning_rate = 2e-4
    learning_algorithm = "eprop"

    # Network parameters
    dynamic_type = nt.SpyLIFLayerLPF
    n_steps = 100
    n_hidden_neurons = 200
    dt = 1e-3

    force_overwrite = True

    checkpoint_folder = f"./checkpoints/heidelberg_{learning_algorithm}_{dynamic_type.__name__}"
    checkpoint_manager = CheckpointManager(
        # checkpoint_folder, metric="val_accuracy", minimise_metric=False, save_best_only=True,
        checkpoint_folder, metric="train_loss", minimise_metric=True, save_best_only=True,
        start_save_at=int(n_iterations * 0.9)
    )
    dataloaders = get_dataloaders(
        as_one_hot=True,
        batch_size=batch_size,
        train_val_split_ratio=1.0,
        pin_memory=True,
    )
    network = SequentialRNN(
        layers=[
            dynamic_type(
                input_size=nt.Size(
                    [
                        Dimension(None, DimensionProperty.TIME),
                        Dimension(dataloaders["test"].dataset.n_units, DimensionProperty.NONE)
                    ]
                ),
                output_size=n_hidden_neurons,
                use_recurrent_connection=False,
                spike_func=HeavisideSigmoidApprox,
                dt=dt,
            ),
            SpyLILayer(
                dt=dt,
                output_size=dataloaders["test"].dataset.n_classes,
                use_bias=True,
                activation="LogSoftmax",
            ),
        ],
        name=f"heidelberg_network",
        hh_memory_size=1,
        checkpoint_folder=checkpoint_folder,
    ).build()
    callbacks = [
        checkpoint_manager,
        nt.Eprop(
            params_lr=learning_rate,
            output_params_lr=2*learning_rate,
            criterion=nt.losses.NLLLoss(target_as_one_hot=True),
            feedbacks_gen_strategy="xavier_normal"
        ),
    ]
    trainer = ClassificationTrainer(
        model=network,
        predict_method="get_last_prediction",
        callbacks=callbacks,
        verbose=True,
    )
    print(trainer)
    training_history = trainer.train(
        dataloaders["train"],
        # dataloaders["val"],
        n_iterations=n_iterations,
        load_checkpoint_mode=LoadCheckpointMode.LAST_ITR,
        force_overwrite=force_overwrite,
    )
    training_history.plot(save_path=f"heidelberg_results/figures/tr_history.png", show=False)
    checkpoint = network.load_checkpoint(
        checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.BEST_ITR, verbose=True
    )
    predictions = {
        k: ClassificationMetrics.compute_y_true_y_pred(network, dataloader, verbose=True, desc=f"{k} predictions")
        for k, dataloader in dataloaders.items()
    }
    accuracies = {
        k: ClassificationMetrics.accuracy(network, y_true=y_true, y_pred=y_pred)
        for k, (y_true, y_pred) in predictions.items()
    }
    precisions = {
        k: ClassificationMetrics.precision(network, y_true=y_true, y_pred=y_pred)
        for k, (y_true, y_pred) in predictions.items()
    }
    recalls = {
        k: ClassificationMetrics.recall(network, y_true=y_true, y_pred=y_pred)
        for k, (y_true, y_pred) in predictions.items()
    }
    f1s = {
        k: ClassificationMetrics.f1(network, y_true=y_true, y_pred=y_pred)
        for k, (y_true, y_pred) in predictions.items()
    }
    results = OrderedDict(
        dict(
            network=str(network),
            accuracies=accuracies,
            precisions=precisions,
            recalls=recalls,
            f1s=f1s,
        )
    )
    with open(f"heidelberg_results/trainer_repr.txt", "w+") as f:
        f.write(repr(trainer))

    json.dump(results, open(f"heidelberg_results/results.json", "w+"), indent=4)
    pprint.pprint(results, indent=4)


