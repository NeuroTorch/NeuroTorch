import unittest

import torch

from neurotorch.modules import ALIFLayer, LIFLayer, LILayer
from neurotorch.modules import SequentialModel, BaseLayer
from neurotorch import Dimension, DimensionProperty


class TestSequential(unittest.TestCase):
	def test_init_base(self):
		model_name = "input"
		model = SequentialModel(
			layers=[
				BaseLayer(Dimension(10, DimensionProperty.NONE), 2, name=model_name),
			]
		)
		self.assertEqual(len(model.input_layers), 0)
		self.assertEqual(len(model.hidden_layers), 0)
		self.assertEqual(len(model.output_layers), 1)
		self.assertEqual(model_name in model.output_layers, True)
		self.assertEqual(model.output_layers[model_name].name, model_name)

	def test_init_sizes_specified(self):
		model = SequentialModel(
			layers=[
				BaseLayer(Dimension(12, DimensionProperty.NONE), 256, name="input"),
				BaseLayer(Dimension(256, DimensionProperty.NONE), 128, name="hidden"),
				BaseLayer(Dimension(128, DimensionProperty.NONE), 10, name="output"),
			]
		)
		self.assertEqual(len(model.input_layers), 1)
		self.assertEqual(len(model.hidden_layers), 1)
		self.assertEqual(len(model.output_layers), 1)

		self.assertEqual(int(list(model.input_sizes.values())[0]), 12)
		self.assertEqual(int(list(model.output_sizes.values())[0]), 10)

		self.assertEqual("input" in model.input_layers, True)
		self.assertEqual(model.hidden_layers[0].name, "hidden")
		self.assertEqual("output" in model.output_layers, True)

		self.assertEqual(int(model.input_layers["input"].input_size), 12)
		self.assertEqual(int(model.input_layers["input"].output_size), 256)
		self.assertEqual(int(model.hidden_layers[0].input_size), 256)
		self.assertEqual(int(model.hidden_layers[0].output_size), 128)
		self.assertEqual(int(model.output_layers["output"].input_size), 128)
		self.assertEqual(int(model.output_layers["output"].output_size), 10)

	def test_init_hidden_sizes_unspecified(self):
		model = SequentialModel(
			layers=[
				BaseLayer(Dimension(12, DimensionProperty.NONE), name="input"),
				BaseLayer(name="hidden"),
				BaseLayer(output_size=10, name="output"),
			]
		)
		self.assertEqual(len(model.input_layers), 1)
		self.assertEqual(len(model.hidden_layers), 1)
		self.assertEqual(len(model.output_layers), 1)

		self.assertEqual(int(list(model.input_sizes.values())[0]), 12)
		self.assertEqual(int(list(model.output_sizes.values())[0]), 10)

		self.assertEqual("input" in model.input_layers, True)
		self.assertEqual(model.hidden_layers[0].name, "hidden")
		self.assertEqual("output" in model.output_layers, True)

		self.assertEqual(model.input_layers["input"].output_size, None)
		self.assertEqual(model.hidden_layers[0].input_size, None)
		self.assertEqual(model.hidden_layers[0].output_size, None)
		self.assertEqual(model.output_layers["output"].input_size, None)

		model.build()

		self.assertEqual(int(model.input_layers["input"].input_size), 12)
		self.assertEqual(int(model.input_layers["input"].output_size), model._default_n_hidden_neurons)
		self.assertEqual(int(model.hidden_layers[0].input_size), model._default_n_hidden_neurons)
		self.assertEqual(int(model.hidden_layers[0].output_size), model._default_n_hidden_neurons)
		self.assertEqual(int(model.output_layers["output"].input_size), model._default_n_hidden_neurons)
		self.assertEqual(int(model.output_layers["output"].output_size), 10)

	def test_init_sizes_unspecified(self):
		model = SequentialModel(
			layers=[
				LIFLayer(name="input"),
				ALIFLayer(name="hidden"),
				LILayer(output_size=10, name="output"),
			]
		)
		self.assertEqual(len(model.input_layers), 1)
		self.assertEqual(len(model.hidden_layers), 1)
		self.assertEqual(len(model.output_layers), 1)

		self.assertEqual(list(model.input_sizes.values())[0], None)
		self.assertEqual(int(list(model.output_sizes.values())[0]), 10)

		self.assertEqual("input" in model.input_layers, True)
		self.assertEqual(model.hidden_layers[0].name, "hidden")
		self.assertEqual("output" in model.output_layers, True)

		self.assertEqual(model.input_layers["input"].output_size, None)
		self.assertEqual(model.hidden_layers[0].input_size, None)
		self.assertEqual(model.hidden_layers[0].output_size, None)
		self.assertEqual(model.output_layers["output"].input_size, None)

		model(torch.randn(1, 12))

		self.assertEqual(int(model.input_layers["input"].input_size), 12)
		self.assertEqual(int(model.input_layers["input"].output_size), model._default_n_hidden_neurons)
		self.assertEqual(int(model.hidden_layers[0].input_size), model._default_n_hidden_neurons)
		self.assertEqual(int(model.hidden_layers[0].output_size), model._default_n_hidden_neurons)
		self.assertEqual(int(model.output_layers["output"].input_size), model._default_n_hidden_neurons)
		self.assertEqual(int(model.output_layers["output"].output_size), 10)

	def test_init_sizes_unspecified_in_dict(self):
		model = SequentialModel(
			layers=[
				LIFLayer(name="input"),
				ALIFLayer(name="hidden"),
				LILayer(output_size=10, name="output"),
			]
		)
		self.assertEqual(len(model.input_layers), 1)
		self.assertEqual(len(model.hidden_layers), 1)
		self.assertEqual(len(model.output_layers), 1)

		self.assertEqual(list(model.input_sizes.values())[0], None)
		self.assertEqual(int(list(model.output_sizes.values())[0]), 10)

		self.assertEqual("input" in model.input_layers, True)
		self.assertEqual(model.hidden_layers[0].name, "hidden")
		self.assertEqual("output" in model.output_layers, True)

		self.assertEqual(model.input_layers["input"].output_size, None)
		self.assertEqual(model.hidden_layers[0].input_size, None)
		self.assertEqual(model.hidden_layers[0].output_size, None)
		self.assertEqual(model.output_layers["output"].input_size, None)

		model({"input": torch.randn(1, 12)})

		self.assertEqual(int(model.input_layers["input"].input_size), 12)
		self.assertEqual(int(model.input_layers["input"].output_size), model._default_n_hidden_neurons)
		self.assertEqual(int(model.hidden_layers[0].input_size), model._default_n_hidden_neurons)
		self.assertEqual(int(model.hidden_layers[0].output_size), model._default_n_hidden_neurons)
		self.assertEqual(int(model.output_layers["output"].input_size), model._default_n_hidden_neurons)
		self.assertEqual(int(model.output_layers["output"].output_size), 10)

	def test_init_input_dict(self):
		model = SequentialModel(
			layers=[
				{"input_0": LIFLayer(6, name="input_0"), "input_1": LIFLayer(12, name="input_1")},
				ALIFLayer(name="hidden"),
				LILayer(output_size=10, name="output"),
			],
		)
		self.assertEqual(len(model.input_layers), 2)
		self.assertEqual(len(model.hidden_layers), 1)
		self.assertEqual(len(model.output_layers), 1)

		self.assertEqual({k: int(v) for k, v in model.input_sizes.items()}, {"input_0": 6, "input_1": 12})
		self.assertEqual(int(list(model.output_sizes.values())[0]), 10)

		self.assertEqual(model.hidden_layers[0].name, "hidden")
		self.assertEqual("output" in model.output_layers, True)

		for k, v in model.input_layers.items():
			self.assertEqual(v.output_size, None)
		self.assertEqual(model.hidden_layers[0].input_size, None)
		self.assertEqual(model.hidden_layers[0].output_size, None)
		self.assertEqual(model.output_layers["output"].input_size, None)

		model.build()

		for k, v in model.input_layers.items():
			self.assertEqual(int(v.output_size), model._default_n_hidden_neurons)
		self.assertEqual(int(model.hidden_layers[0].input_size), model._default_n_hidden_neurons * len(model.input_layers))
		self.assertEqual(int(model.hidden_layers[0].output_size), model._default_n_hidden_neurons)
		self.assertEqual(int(model.output_layers["output"].input_size), model._default_n_hidden_neurons)
		self.assertEqual(int(model.output_layers["output"].output_size), 10)

	def test_init_output_dict(self):
		model = SequentialModel(
			layers=[
				{"input_0": LIFLayer(6), "input_1": LIFLayer(12)},
				ALIFLayer(name="hidden"),
				{"output_0": LILayer(output_size=10), "output_1": LILayer(output_size=16)},
			],
		)
		self.assertEqual(len(model.input_layers), 2)
		self.assertEqual(len(model.hidden_layers), 1)
		self.assertEqual(len(model.output_layers), 2)

		self.assertEqual({k: int(v) for k, v in model.input_sizes.items()}, {"input_0": 6, "input_1": 12})
		self.assertEqual({k: int(v) for k, v in model.output_sizes.items()}, {"output_0": 10, "output_1": 16})
		self.assertEqual(model.hidden_layers[0].name, "hidden")

		for k, v in model.input_layers.items():
			self.assertEqual(v.output_size, None)
		self.assertEqual(model.hidden_layers[0].input_size, None)
		self.assertEqual(model.hidden_layers[0].output_size, None)
		for k, v in model.output_layers.items():
			self.assertEqual(v.input_size, None)

		model({"input_0": torch.randn(1, 6), "input_1": torch.randn(1, 12)})

		for k, v in model.input_layers.items():
			self.assertEqual(int(v.output_size), model._default_n_hidden_neurons)
		self.assertEqual(int(model.hidden_layers[0].input_size), model._default_n_hidden_neurons * len(model.input_layers))
		self.assertEqual(int(model.hidden_layers[0].output_size), model._default_n_hidden_neurons)
		for k, v in model.output_layers.items():
			self.assertEqual(int(v.input_size), model._default_n_hidden_neurons)

	def test_init_input_output_list(self):
		model = SequentialModel(
			layers=[
				[LIFLayer(6, name="input_0"), LIFLayer(12, name="input_1")],
				ALIFLayer(name="hidden"),
				[LILayer(output_size=10), LILayer(output_size=16)],
			],
		)
		self.assertEqual(len(model.input_layers), 2)
		self.assertEqual(len(model.hidden_layers), 1)
		self.assertEqual(len(model.output_layers), 2)

		self.assertEqual({k: int(v) for k, v in model.input_sizes.items()}, {"input_0": 6, "input_1": 12})
		self.assertEqual({k: int(v) for k, v in model.output_sizes.items()}, {"output_0": 10, "output_1": 16})
		self.assertEqual(model.hidden_layers[0].name, "hidden")

		for k, v in model.input_layers.items():
			self.assertEqual(v.output_size, None)
		self.assertEqual(model.hidden_layers[0].input_size, None)
		self.assertEqual(model.hidden_layers[0].output_size, None)
		for k, v in model.output_layers.items():
			self.assertEqual(v.input_size, None)

		model.build()

		for k, v in model.input_layers.items():
			self.assertEqual(int(v.output_size), model._default_n_hidden_neurons)
		self.assertEqual(int(model.hidden_layers[0].input_size), model._default_n_hidden_neurons * len(model.input_layers))
		self.assertEqual(int(model.hidden_layers[0].output_size), model._default_n_hidden_neurons)
		for k, v in model.output_layers.items():
			self.assertEqual(int(v.input_size), model._default_n_hidden_neurons)

	def test_init_input_output_list_names_unspecified(self):
		model = SequentialModel(
			layers=[
				[LIFLayer(6), LIFLayer(12)],
				ALIFLayer(),
				ALIFLayer(),
				[LILayer(output_size=10), LILayer(output_size=16)],
			],
		)
		self.assertEqual(len(model.input_layers), 2)
		self.assertEqual(len(model.hidden_layers), 2)
		self.assertEqual(len(model.output_layers), 2)

		self.assertEqual({k: int(v) for k, v in model.input_sizes.items()}, {"input_0": 6, "input_1": 12})
		self.assertEqual({k: int(v) for k, v in model.output_sizes.items()}, {"output_0": 10, "output_1": 16})
		self.assertEqual(len([hh.name for hh in model.hidden_layers]), len(set([hh.name for hh in model.hidden_layers])))

		for k, v in model.input_layers.items():
			self.assertEqual(v.output_size, None)
		for v in model.hidden_layers:
			self.assertEqual(v.input_size, None)
			self.assertEqual(v.output_size, None)
		for k, v in model.output_layers.items():
			self.assertEqual(v.input_size, None)

		model.build()

		for k, v in model.input_layers.items():
			self.assertEqual(int(v.output_size), model._default_n_hidden_neurons)

		for i, v in enumerate(model.hidden_layers):
			self.assertEqual(
				int(v.input_size), model._default_n_hidden_neurons * (len(model.input_layers) if i == 0 else 1)
			)
			self.assertEqual(int(v.output_size), model._default_n_hidden_neurons)
		for k, v in model.output_layers.items():
			self.assertEqual(int(v.input_size), model._default_n_hidden_neurons)


