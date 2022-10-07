<p align="center" width="100%">
    <img width="40%" src="images/neurotorch.svg">
</p>

[![Star on GitHub](https://img.shields.io/github/stars/NeuroTorch/NeuroTorch.svg?style=social)](https://github.com/NeuroTorch/NeuroTorch/stargazers)
[![Python 3.6](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

![Tests Workflow](https://github.com/NeuroTorch/NeuroTorch/actions/workflows/tests.yml/badge.svg)
![Dist Workflow](https://github.com/NeuroTorch/NeuroTorch/actions/workflows/build_dist.yml/badge.svg)
![Doc Workflow](https://github.com/NeuroTorch/NeuroTorch/actions/workflows/docs.yml/badge.svg)
![Publish Workflow](https://github.com/NeuroTorch/NeuroTorch/actions/workflows/publish.yml/badge.svg)


# Description

It's time to bring deep learning and neuroscience together. In this library, we offer machine learning 
tools to neuroscientists and we offer neuroscience tools to computer scientists. These two domains were created 
to be one.

### Current Version (v0.0.1-alpha)

What can we do with NeuroTorch in the current version? 
- Image classification with spiking networks.
- Classification of spiking time series with spiking networks.
- Time series classification with spiking or Wilson-Cowan.
- Reconstruction/Prediction of time series with Wilson-Cowan;
- Reconstruction/Prediction of continuous time series with spiking networks.  
- Backpropagation Through Time.
- Anything you are able to do using the modules already created.


### Next Version (v0.0.1-beta)

- Learning Algorithm: FullForce.
- Learning Algorithm: [Eligibility-Propagation](https://doi.org/10.1038/s41467-020-17236-y).

### Upcoming Version (v0.0.1)
- Reinforcement Learning.


NeuroTorch is developed to be easy to use, so that you can do simple things in a few lines of code. 
Moreover NeuroTorch is modular so you can adapt it to your needs relatively quickly. Thanks and stay tuned, 
because more is coming!  


This package is part of a postgraduate research project realized by [Jérémie Gince](https://github.com/JeremieGince) and supervised by [Simon Hardy]() and [Patrick Desrosiers](https://github.com/pdesrosiers). Our work was supported by: (1) [UNIQUE](https://www.unique.quebec/home), a FRQNT-funded research center, (2) the [Sentinelle Nord](https://sentinellenord.ulaval.ca/en) program of Université Laval, funded by the Canada First Research Excellence Fund, and (3) [NSERC](https://www.nserc-crsng.gc.ca).




# Important Links

  - Documentation at [https://NeuroTorch.github.io/NeuroTorch/](https://NeuroTorch.github.io/NeuroTorch/).
  - Github at [https://github.com/NeuroTorch/NeuroTorch/](https://github.com/NeuroTorch/NeuroTorch/).


# Installation

## Using pip

```bash
pip install neurotorch
```


## With wheel:

   1. Download the .whl file [here](https://github.com/NeuroTorch/NeuroTorch/tree/main/dist);
   2. Copy the path of this file on your computer;
   3. pip install it with ``` pip install [path].whl ```

## With pip+git:

```bash
pip install git+https://github.com/NeuroTorch/NeuroTorch
```

# Tutorials / Applications

See the readme of the tutorials folder [here](tutorials/README.md).

## Image classification with spiking networks (Mnist/Fashion-Mnist)

- Tutorial: [Jupyter Notebook](tutorials/mnist/tutorial.ipynb).
- Project: [Repository](https://github.com/NeuroTorch/MnistClassification_NeuroTorch).

## Classification of spiking time series (Heidelberg)

- Tutorial: [Jupyter Notebook](tutorials/heidelberg/tutorial.ipynb).
- Project: [Repository](https://github.com/NeuroTorch/HeidelbergClassification_NeuroTorch).

## Time series classification with spiking networks

**Sorry, it's a work in progress, so it's not publish yet.**

- Tutorial: [Jupyter Notebook](tutorials/time_series_forecasting_spiking/tutorial.ipynb).
- Project: [Repository](https://github.com/NeuroTorch/SNN_TS_Forecasting_NeuroTorch).

## Time series classification with Wilson-Cowan

- Tutorial: [Jupyter Notebook](tutorials/time_series_forecasting_wilson_cowan/tutorial.ipynb).


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
		nt.LIFLayer(
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
	callbacks=[
        checkpoint_manager,
    ],
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


# Found a bug or have a feature request?
- [Click here to create a new issue.](https://github.com/NeuroTorch/NeuroTorch/issues/new)


# Thanks
- [Anthony Drouin](https://github.com/AnthoDrouin) who helped develop the Wilson-Cowan application during his 2022 summer internship.
- [Antoine Légaré](https://github.com/AntoineLegare) who made the awesome [logo](images/neurotorch.svg) of NeuroTorch.
- To my dog Chewy who has been a great help during the whole development.

# License
[Apache License 2.0](LICENSE)

# Citation
```
@misc{Gince2022,
  title={NeuroTorch: Deep Learning Python Library for Machine Learning and Neuroscience.},
  author={Jérémie Gince},
  year={2022},
  publisher={Université Laval},
  url={https://github.com/NeuroTorch},
}
```

---------------------------------------------------------------------------
