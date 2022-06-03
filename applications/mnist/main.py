import os
import pprint
from typing import Any, Dict

import psutil

from applications.mnist.dataset import get_dataloaders, DatasetId
from neurotorch import Dimension, DimensionProperty
from neurotorch.callbacks import CheckpointManager, LoadCheckpointMode
from neurotorch.metrics import ClassificationMetrics
from neurotorch.modules import SequentialModel, ALIFLayer, LILayer
from neurotorch.trainers import ClassificationTrainer
from neurotorch.utils import hash_params


def train_with_params(params: Dict[str, Any], data_folder="tr_results", verbose=False):
	checkpoints_name = str(hash_params(params))
	checkpoint_folder = f"{data_folder}/{checkpoints_name}"
	os.makedirs(checkpoint_folder, exist_ok=True)

	dataloaders = get_dataloaders(
		dataset_id=params["dataset_id"],
		batch_size=256,
		n_steps=params["n_steps"],
		to_spikes_use_periods=params["to_spikes_use_periods"],
		train_val_split_ratio=params.get("train_val_split_ratio", 0.85),
		nb_workers=psutil.cpu_count(logical=False),
	)
	network = SequentialModel(
		layers=[
			ALIFLayer(input_size=Dimension(28 * 28, DimensionProperty.NONE), output_size=128),
			LILayer(input_size=128, output_size=10),
		],
		name="mnist_network",
		checkpoint_folder=checkpoint_folder,
	)
	# save_params(params, os.path.join(checkpoint_folder, "params.pkl"))
	trainer = ClassificationTrainer(
		model=network,
		callbacks=CheckpointManager(checkpoint_folder)
	)
	trainer.train(
		dataloaders["train"],
		dataloaders["val"],
		n_iterations=params.get("n_iterations", 15),
		load_checkpoint_mode=LoadCheckpointMode.LAST_ITR,
		force_overwrite=False,
		verbose=verbose,
	)
	# network.load_checkpoint(LoadCheckpointMode.BEST_EPOCH)
	return dict(
		network=network,
		accuracies={
			k: ClassificationMetrics.accuracy(network, dataloaders[k], verbose=True, desc=k)
			for k in dataloaders
		},
		checkpoints_name=checkpoints_name,
	)


if __name__ == '__main__':
	results = train_with_params(
		{
			"dataset_id": DatasetId.MNIST,
			"to_spikes_use_periods": False,
			# "learn_beta": True,
			"n_iterations": 30,
			"n_steps": 2,
			"train_val_split_ratio": 0.95,
		},
		verbose=True,
	)
	pprint.pprint(results, indent=4)
