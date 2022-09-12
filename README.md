<p align="center" width="100%">
    <img width="75%" src="images/neurotorch_logo.png">
</p>


# Description

It's time to bring together deep learning and neuroscience together. In this library, we offer machine learning 
tools to neuroscientists and we offer neuroscience tools to computer scientists. These two domains were created 
to be one.

# Important Links

  - Documentation at [https://jeremiegince.github.io/NeuroTorch/](https://jeremiegince.github.io/NeuroTorch/).
  - Github at [https://github.com/JeremieGince/NeuroTorch/](https://github.com/JeremieGince/NeuroTorch/).


# Installation

## With wheel:

   1. Download the .whl file [here](https://github.com/JeremieGince/NeuroTorch/tree/main/dist/NeuroTorch-0.0.0.1-py3-none-any.whl);
   2. Copy the path of this file on your computer;
   3. pip install it with ``` pip install [path].whl ```

## With pip+git:

```bash
pip install git+https://github.com/JeremieGince/NeuroTorch
```


# Quick usage preview

```python
import neurotorch as nt
import torch
import pprint

n_hidden_neurons = 128
checkpoint_folder = "./checkpoints/checkpoint_000"
checkpoint_manager = nt.CheckpointManager(checkpoint_folder)
dataloaders = get_dataloaders(
	batch_size=256,
	train_val_split_ratio=0.95,
)

network = nt.SequentialModel(
	layers=[
		nt.SpyLIFLayer(
			input_size=nt.Size([
				nt.Dimension(None, nt.DimensionProperty.TIME),
				nt.Dimension(dataloaders["test"].dataset.n_units, nt.DimensionProperty.NONE)
			]),
			output_size=n_hidden_neurons,
			use_recurrent_connection=True,
		),
		nt.SpyLILayer(output_size=dataloaders["test"].dataset.n_classes),
	],
	name=f"Network",
	checkpoint_folder=checkpoint_folder,
).build()

trainer = nt.ClassificationTrainer(
	model=network,
	optimizer=torch.optim.Adam(network.parameters(), lr=1e-3),
	callbacks=checkpoint_manager,
	verbose=True,
)
training_history = trainer.train(
	dataloaders["train"],
	dataloaders["val"],
	n_iterations=100,
	load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR,
)
training_history.plot(show=True)

network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, nt.LoadCheckpointMode.BEST_ITR, verbose=True)
predictions = {
	k: nt.metrics.ClassificationMetrics.compute_y_true_y_pred(network, dataloader, verbose=True, desc=f"{k} predictions")
	for k, dataloader in dataloaders.items()
}
accuracies = {
	k: nt.metrics.ClassificationMetrics.accuracy(network, y_true=y_true, y_pred=y_pred)
	for k, (y_true, y_pred) in predictions.items()
}
pprint.pprint(accuracies)
```




---------------------------------------------------------------------------
