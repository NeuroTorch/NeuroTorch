<div style="text-align:center"><img src="images/neurotorch.svg" width="40%" /></div>


[![Star on GitHub](https://img.shields.io/github/stars/NeuroTorch/NeuroTorch.svg?style=social)](https://github.com/NeuroTorch/NeuroTorch/stargazers)
[![Python 3.6](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

![Tests Workflow](https://github.com/NeuroTorch/NeuroTorch/actions/workflows/tests.yml/badge.svg)
![Dist Workflow](https://github.com/NeuroTorch/NeuroTorch/actions/workflows/build_dist.yml/badge.svg)
![Doc Workflow](https://github.com/NeuroTorch/NeuroTorch/actions/workflows/docs.yml/badge.svg)
![Publish Workflow](https://github.com/NeuroTorch/NeuroTorch/actions/workflows/publish.yml/badge.svg)




# 1. Description

It's time to bring deep learning and neuroscience together. With this library, we offer machine learning 
tools to neuroscientists and we offer neuroscience tools to computer scientists. These two domains were created 
to be one!


NeuroTorch was developed to be easy to use and you can do simple things with few lines of code. 
Moreover, NeuroTorch is modular so you can adapt it to your needs relatively quickly. Thanks and stay tuned, 
because more is coming!


### Current Version

What can be done with NeuroTorch in the current version? 
- Image classification with spiking networks.
- Classification of spiking time series with spiking networks.
- Time series classification with spiking or Wilson-Cowan.
- Reconstruction/Prediction of time series with Wilson-Cowan;
- Reconstruction/Prediction of continuous time series with spiking networks.  
- Backpropagation Through Time (BPTT).
- Truncated-Backpropagation-Through-Time (TBPTT).
- Learning Algorithm: [Eligibility-Propagation](https://doi.org/10.1038/s41467-020-17236-y).
- Anything you are able to do using the modules already created.
- Reinforcement Learning.


### Next Versions
- Learning Algorithm: RLS (Recursive Least Squares).




# 2. Installation



| Method     | Commands                                                                                                                                                                       |
|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **PyPi**   | `pip install neurotorch`                                                                                                                                                       |
| **source** | `pip install git+https://github.com/NeuroTorch/NeuroTorch`                                                                                                                     |
| **wheel**  | 1.Download the .whl file [here](https://github.com/NeuroTorch/NeuroTorch/tree/main/dist);<br> 2. Copy the path of this file on your computer; <br> 3. `pip install [path].whl` |


### 2.1 Last unstable version
To install the last unstable version, you can install it by downloading the last version of the .whl file
and following the instructions above.




# 3. Tutorials / Applications

See the readme of the tutorials folder [here](tutorials/README.md).

| Tutorial                                                                          | Project                                                                         | Description                                                                                                                         |
|-----------------------------------------------------------------------------------|---------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| [Jupyter Notebook](tutorials/mnist/tutorial.ipynb)                                | [Repository](https://github.com/NeuroTorch/MnistClassification_NeuroTorch)      | Image classification with spiking networks (Mnist/Fashion-Mnist).                                                                   |
| [Jupyter Notebook](tutorials/heidelberg/tutorial.ipynb)                           | [Repository](https://github.com/NeuroTorch/HeidelbergClassification_NeuroTorch) | Time series classification with spiking networks (Heidelberg).                                                                      |
| [Jupyter Notebook](tutorials/time_series_forecasting_spiking/tutorial.ipynb)      | [Repository](https://github.com/NeuroTorch/SNN_TS_Forecasting_NeuroTorch)       | Time series forecasting with spiking networks (Neuronal activity) <br> **Sorry, it's a work in progress, so it's not publish yet.** | 
| [Jupyter Notebook](tutorials/time_series_forecasting_wilson_cowan/tutorial.ipynb) | Null                                                                            | Time series forecasting with Wilson-Cowan (Neuronal activity).                                                                      |




# 4. Quick usage preview

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

network = nt.SequentialRNN(
	layers=[
		nt.LIFLayer(
			input_size=nt.Size(
				[
					nt.Dimension(None, nt.DimensionProperty.TIME),
					nt.Dimension(dataloaders["test"].dataset.n_units, nt.DimensionProperty.NONE)
				]
			),
			output_size=n_hidden_neurons,
			use_recurrent_connection=True,
		),
		nt.SpyLILayer(output_size=dataloaders["test"].dataset.n_classes),
	],
	name=f"Network",
	checkpoint_folder=checkpoint_folder,
).build()

learning_algorithm = nt.BPTT(optimizer=torch.optim.Adam(network.parameters(), lr=1e-3))
trainer = nt.ClassificationTrainer(
	model=network,
	callbacks=[checkpoint_manager, learning_algorithm],
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
	k: nt.metrics.ClassificationMetrics.compute_y_true_y_pred(
		network, dataloader, verbose=True, desc=f"{k} predictions"
		)
	for k, dataloader in dataloaders.items()
}
accuracies = {
	k: nt.metrics.ClassificationMetrics.accuracy(network, y_true=y_true, y_pred=y_pred)
	for k, (y_true, y_pred) in predictions.items()
}
pprint.pprint(accuracies)
```



# 5. Why NeuroTorch?
On the one hand, neuroscientists are increasingly using machine learning (ML) without necessarily having the 
expertise to create training pipelines. On the other hand, most ML experts lack the neuroscience background to 
implement biologically inspired models. There is thus a need for a tool providing a complete ML pipeline with 
features originating from neuroscience while using a simple and intuitive interface. 

The goal of this work is to provide a Python package, NeuroTorch, offering a flexible and intuitive training pipeline 
together with biologically-constrained neuronal dynamics. This tool will include several learning strategies highly 
used in both ML and neuroscience to ensure that both fields can benefit from the package.




# 6. Similar work

* [Norse](https://github.com/norse/norse) is a highly optimized spiking neural network library for PyTorch. In fact, 
this library seems to be very similar to NeuroTorch at first glance. However, the main difference is that NeuroTorch
is focused on the development of learning algorithms for spiking neural networks and other bio-inspired dynamics like 
Wilson-Cowan, while Norse is focused on the development of spiking neural networks layers itself. In addition,  
NeuroTorch will soon allow to easily use modules from Norse.
* [SpyTorch](https://github.com/fzenke/spytorch) presents a set of tutorials for training SNNs with the surrogate 
gradient approach SuperSpike by [F. Zenke, and S. Ganguli (2017)](https://arxiv.org/abs/1705.11146). In fact,
the prefix 'Spy' of certain layers in NeuroTorch is a reference to SpyTorch.
* [PySNN](https://github.com/BasBuller/PySNN/) is a PyTorch extension similar to [Norse](https://github.com/norse/norse).
* [Pytorch Lightning](https://github.com/Lightning-AI/lightning) is a deep learning framework to train, deploy, and 
  ship AI products Lightning fast. 
* [Poutyne](https://github.com/GRAAL-Research/poutyne) is a simplified framework for PyTorch and handles much of the 
  boilerplating code needed to train classical neural networks.





# 7. About

This package is part of a postgraduate research project realized by [Jérémie Gince](https://github.com/JeremieGince) 
and supervised by [Simon V Hardy]() and [Patrick Desrosiers](https://github.com/pdesrosiers). 
Our work was supported by: (1) [UNIQUE](https://www.unique.quebec/home), a FRQNT-funded research center, (2) 
the [Sentinelle Nord](https://sentinellenord.ulaval.ca/en) program of Université Laval, funded by the Canada 
First Research Excellence Fund, and (3) [NSERC](https://www.nserc-crsng.gc.ca).




# 8. Important Links
  - Documentation at [https://NeuroTorch.github.io/NeuroTorch/](https://NeuroTorch.github.io/NeuroTorch/).
  - Github at [https://github.com/NeuroTorch/NeuroTorch/](https://github.com/NeuroTorch/NeuroTorch/).




# 9. Found a bug or have a feature request?
- [Click here to create a new issue.](https://github.com/NeuroTorch/NeuroTorch/issues/new)



# 10. Thanks
- [Anthony Drouin](https://github.com/AnthoDrouin) who helped develop the Wilson-Cowan application during his 2022 
  summer internship and who is now a collaborator of the project.
- [Antoine Légaré](https://github.com/AntoineLegare) and [Thomas Charland]() who made the awesome 
  [logo](images/neurotorch.svg) of NeuroTorch.
- To my dog Chewy who has been a great help during the whole development.



# 11. License
[Apache License 2.0](LICENSE)



# 12. Citation
```
@misc{Gince2022,
  title={NeuroTorch: A Python library for machine learning and neuroscience.},
  author={Jérémie Gince},
  year={2022},
  publisher={Université Laval},
  url={https://github.com/NeuroTorch},
}
```


