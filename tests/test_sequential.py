import unittest
from typing import Iterable

import numpy as np
import torch
from torchvision.transforms import Compose

from neurotorch.modules import ALIFLayer, LIFLayer, LILayer
from neurotorch.modules import SequentialModel, BaseLayer
from neurotorch import Dimension, DimensionProperty
from neurotorch.utils import ravel_compose_transforms


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
		self.assertTrue(model.is_built)

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
		self.assertTrue(model.is_built)

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
		self.assertTrue(model.is_built)

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
		self.assertTrue(model.is_built)

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
		self.assertTrue(model.is_built)

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
		self.assertTrue(model.is_built)

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
		self.assertTrue(model.is_built)

		for k, v in model.input_layers.items():
			self.assertEqual(int(v.output_size), model._default_n_hidden_neurons)

		for i, v in enumerate(model.hidden_layers):
			self.assertEqual(
				int(v.input_size), model._default_n_hidden_neurons * (len(model.input_layers) if i == 0 else 1)
			)
			self.assertEqual(int(v.output_size), model._default_n_hidden_neurons)
		for k, v in model.output_layers.items():
			self.assertEqual(int(v.input_size), model._default_n_hidden_neurons)

	def test_init_device_specified(self):
		# Test that the model is initialized with the specified device
		model = SequentialModel(
			layers=[
				BaseLayer(Dimension(10, DimensionProperty.NONE), 2, name="input"),
			],
			device=torch.device("cpu"),
		)
		model.build()
		self.assertEqual(model.device, torch.device("cpu"))
		for layer in model.get_all_layers():
			self.assertEqual(layer.device, torch.device("cpu"))

		# Test that the model is initialized with the specified device if it is specified in the layer
		model = SequentialModel(
			layers=[
				BaseLayer(Dimension(10, DimensionProperty.NONE), 2, name="input", device=torch.device("cpu")),
			],
			device=torch.device("cuda"),
		)
		model.build()
		self.assertEqual(model.device, torch.device("cuda"))
		for layer in model.get_all_layers():
			self.assertEqual(layer.device, torch.device("cuda"))

		# Test that the model is initialized with the specified device if it is specified in the layer
		model = SequentialModel(
			layers=[
				BaseLayer(Dimension(10, DimensionProperty.NONE), 2, name="input", device=torch.device("cuda")),
			],
			device=torch.device("cpu"),
		)
		model.build()
		self.assertEqual(model.device, torch.device("cpu"))
		for layer in model.get_all_layers():
			self.assertEqual(layer.device, torch.device("cpu"))

		# Test that the model is initialized with the specified device if it is specified in the layer
		model = SequentialModel(
			layers=[
				BaseLayer(Dimension(10, DimensionProperty.NONE), 10, device=torch.device("cuda")),
				BaseLayer(Dimension(10, DimensionProperty.NONE), 10, device=torch.device("cuda")),
				BaseLayer(Dimension(10, DimensionProperty.NONE), 10, device=torch.device("cuda")),
				BaseLayer(Dimension(10, DimensionProperty.NONE), 10, device=torch.device("cuda")),
				BaseLayer(Dimension(10, DimensionProperty.NONE), 10, device=torch.device("cuda")),
			],
			device=torch.device("cpu"),
		)
		model.build()
		self.assertEqual(model.device, torch.device("cpu"))
		for layer in model.get_all_layers():
			self.assertEqual(layer.device, torch.device("cpu"))

	def test_format_hidden_outputs_traces(self):
		data = torch.ones((32, 2))
		time_steps = 10
		hh_states = {
			'0': [(0 * data, 1 * data, 2 * data) for _ in range(time_steps)]
		}
		hh_states_transposed = {
			'0': tuple([torch.stack(e, dim=1) for e in list(zip(*hh_states['0']))])
		}
		hh_pred = SequentialModel._format_hidden_outputs_traces(hh_states)
		self.assertTrue(all(torch.allclose(x, y) for x, y in zip(hh_states_transposed['0'], hh_pred['0'])))

		hh_states = {
			'0': [(data, ) for _ in range(time_steps)]
		}
		hh_states_transposed = {
			'0': torch.stack([data for _ in range(time_steps)], dim=1)
		}
		hh_pred = SequentialModel._format_hidden_outputs_traces(hh_states)
		self.assertTrue(all(torch.allclose(x, y) for x, y in zip(hh_states_transposed['0'], hh_pred['0'])))

		hh_states = {
			'0': [data for _ in range(time_steps)]
		}
		hh_states_transposed = {
			'0': torch.stack([data for _ in range(time_steps)], dim=1)
		}
		hh_pred = SequentialModel._format_hidden_outputs_traces(hh_states)
		self.assertTrue(torch.allclose(hh_states_transposed['0'], hh_pred['0']))

		hh_states = {
			'0': [(None, ) for _ in range(time_steps)]
		}
		hh_states_transposed = {
			'0': [None for _ in range(time_steps)]
		}
		hh_pred = SequentialModel._format_hidden_outputs_traces(hh_states)
		self.assertEqual(hh_states_transposed['0'],  hh_pred['0'])

		hh_states = {
			'0': [None for _ in range(time_steps)]
		}
		hh_states_transposed = {
			'0': [None for _ in range(time_steps)]
		}
		hh_pred = SequentialModel._format_hidden_outputs_traces(hh_states)
		self.assertEqual(hh_states_transposed['0'], hh_pred['0'])

	def test_init_transforms(self):
		# Test that the model is initialized with the default transforms with one layer
		model = SequentialModel(
			layers=[
				BaseLayer(Dimension(10, DimensionProperty.NONE), 2, name="input"),
			],
		)
		model.build()
		self.assertGreater(len(model.input_transform), 0)
		for key, value in model.input_transform.items():
			self.assertIn(
				key, model.output_layers.keys(),
				f"{key} is not in the output layers {model.output_layers.keys()}"
			)

		# Test that the model is initialized with the default transforms with multiple layers
		model = SequentialModel(
			layers=[
				BaseLayer(Dimension(10, DimensionProperty.NONE), 2),
				BaseLayer(Dimension(10, DimensionProperty.NONE), 2),
			],
		)
		model.build()
		self.assertGreater(len(model.input_transform), 0)
		for key, value in model.input_transform.items():
			self.assertIn(
				key, model.input_layers.keys(),
				f"{key} is not in the output layers {model.input_layers.keys()}"
			)

		def _dummy_transform(x):
			return x

		# Test that the model is initialized with the specified transforms
		model = SequentialModel(
			layers=[
				BaseLayer(Dimension(10, DimensionProperty.NONE), 2),
				BaseLayer(Dimension(10, DimensionProperty.NONE), 2),
			],
			input_transform=[_dummy_transform]
		)
		model.build()
		self.assertEqual(len(model.input_transform), 1)
		for key, value in model.input_transform.items():
			self.assertIn(
				key, model.input_layers.keys(),
				f"{key} is not in the output layers {model.input_layers.keys()}"
			)
			self.assertIn(_dummy_transform, ravel_compose_transforms(value))

	def test_if_grad(self):
		model = SequentialModel(
			layers=[
				LILayer(Dimension(10, DimensionProperty.NONE), 10),
				LILayer(Dimension(10, DimensionProperty.NONE), 2),
			],
		)
		model.build()
		for layer in model.get_all_layers():
			self.assertTrue(layer.forward_weights.requires_grad)
			self.assertTrue(layer.bias_weights.requires_grad)

	def test_call_backward(self):
		model = SequentialModel(
			layers=[
				LILayer(Dimension(10, DimensionProperty.NONE), 10),
				LILayer(Dimension(10, DimensionProperty.NONE), 10),
				LILayer(Dimension(10, DimensionProperty.NONE), 10),
				LILayer(Dimension(10, DimensionProperty.NONE), 10),
				LILayer(Dimension(10, DimensionProperty.NONE), 2),
			],
		)
		model.build()
		for layer in model.get_all_layers():
			self.assertTrue(layer.forward_weights.requires_grad)
			self.assertTrue(layer.bias_weights.requires_grad)

		x = torch.randn(1, 100, 10)
		y, hh = model(x)
		for key, value in y.items():
			self.assertTrue(value.requires_grad)
			self.assertEqual(value.shape, torch.Size([1, 100, 2]))
			value.mean().backward()

		for layer in model.get_all_layers():
			self.assertIsInstance(layer.forward_weights.grad, torch.Tensor)
			self.assertEqual(layer.forward_weights.grad.shape, layer.forward_weights.shape)
			self.assertEqual(layer.forward_weights.grad.device.type, layer.device.type)
			self.assertIsInstance(layer.bias_weights.grad, torch.Tensor)
			self.assertEqual(layer.bias_weights.grad.shape, layer.bias_weights.shape)
			self.assertEqual(layer.bias_weights.grad.device.type, layer.device.type)

	def test_get_prediction_trace_backward(self):
		model = SequentialModel(
			layers=[
				LILayer(Dimension(10, DimensionProperty.NONE), 10),
				LILayer(Dimension(10, DimensionProperty.NONE), 10),
				LILayer(Dimension(10, DimensionProperty.NONE), 10),
				LILayer(Dimension(10, DimensionProperty.NONE), 10),
				LILayer(Dimension(10, DimensionProperty.NONE), 10),
			],
			foresight_time_steps=100,
		)
		model.build()
		for layer in model.get_all_layers():
			self.assertTrue(layer.forward_weights.requires_grad)
			self.assertTrue(layer.bias_weights.requires_grad)

		x = torch.randn(1, 100, 10)
		y = model.get_prediction_trace(x)
		self.assertTrue(y.requires_grad)
		self.assertEqual(y.shape, torch.Size([1, 100, 10]))
		y.mean().backward()

		for layer in model.get_all_layers():
			self.assertIsInstance(layer.forward_weights.grad, torch.Tensor)
			self.assertEqual(layer.forward_weights.grad.shape, layer.forward_weights.shape)
			self.assertEqual(layer.forward_weights.grad.device.type, layer.device.type)
			self.assertIsInstance(layer.bias_weights.grad, torch.Tensor)
			self.assertEqual(layer.bias_weights.grad.shape, layer.bias_weights.shape)
			self.assertEqual(layer.bias_weights.grad.device.type, layer.device.type)

	def test_get_raw_prediction_backward(self):
		model = SequentialModel(
			layers=[
				LILayer(Dimension(10, DimensionProperty.NONE), 10),
				LILayer(Dimension(10, DimensionProperty.NONE), 10),
				LILayer(Dimension(10, DimensionProperty.NONE), 10),
				LILayer(Dimension(10, DimensionProperty.NONE), 10),
				LILayer(Dimension(10, DimensionProperty.NONE), 2),
			],
		)
		model.build()
		for layer in model.get_all_layers():
			self.assertTrue(layer.forward_weights.requires_grad)
			self.assertTrue(layer.bias_weights.requires_grad)

		x = torch.randn(1, 100, 10)
		y, o, hh = model.get_raw_prediction(x)
		for key, value in y.items():
			self.assertTrue(value.requires_grad)
			self.assertEqual(value.shape, torch.Size([1, 2]))
			value.mean().backward()

		for layer in model.get_all_layers():
			self.assertIsInstance(layer.forward_weights.grad, torch.Tensor)
			self.assertEqual(layer.forward_weights.grad.shape, layer.forward_weights.shape)
			self.assertEqual(layer.forward_weights.grad.device.type, layer.device.type)
			self.assertIsInstance(layer.bias_weights.grad, torch.Tensor)
			self.assertEqual(layer.bias_weights.grad.shape, layer.bias_weights.shape)
			self.assertEqual(layer.bias_weights.grad.device.type, layer.device.type)

	def test_get_prediction_proba_backward(self):
		model = SequentialModel(
			layers=[
				LILayer(Dimension(10, DimensionProperty.NONE), 10),
				LILayer(Dimension(10, DimensionProperty.NONE), 10),
				LILayer(Dimension(10, DimensionProperty.NONE), 10),
				LILayer(Dimension(10, DimensionProperty.NONE), 10),
				LILayer(Dimension(10, DimensionProperty.NONE), 2),
			],
		)
		model.build()
		for layer in model.get_all_layers():
			self.assertTrue(layer.forward_weights.requires_grad)
			self.assertTrue(layer.bias_weights.requires_grad)

		x = torch.randn(1, 100, 10)
		y, o, hh = model.get_prediction_proba(x)
		for key, value in y.items():
			self.assertTrue(value.requires_grad)
			self.assertEqual(value.shape, torch.Size([1, 2]))
			value.mean().backward()

		for layer in model.get_all_layers():
			self.assertIsInstance(layer.forward_weights.grad, torch.Tensor)
			self.assertEqual(layer.forward_weights.grad.shape, layer.forward_weights.shape)
			self.assertEqual(layer.forward_weights.grad.device.type, layer.device.type)
			self.assertIsInstance(layer.bias_weights.grad, torch.Tensor)
			self.assertEqual(layer.bias_weights.grad.shape, layer.bias_weights.shape)
			self.assertEqual(layer.bias_weights.grad.device.type, layer.device.type)

	def test_get_prediction_log_proba_backward(self):
		model = SequentialModel(
			layers=[
				LILayer(Dimension(10, DimensionProperty.NONE), 10),
				LILayer(Dimension(10, DimensionProperty.NONE), 10),
				LILayer(Dimension(10, DimensionProperty.NONE), 10),
				LILayer(Dimension(10, DimensionProperty.NONE), 10),
				LILayer(Dimension(10, DimensionProperty.NONE), 2),
			],
		)
		model.build()
		for layer in model.get_all_layers():
			self.assertTrue(layer.forward_weights.requires_grad)
			self.assertTrue(layer.bias_weights.requires_grad)

		x = torch.randn(1, 100, 10)
		y, o, hh = model.get_prediction_log_proba(x)
		for key, value in y.items():
			self.assertTrue(value.requires_grad)
			self.assertEqual(value.shape, torch.Size([1, 2]))
			value.mean().backward()

		for layer in model.get_all_layers():
			self.assertIsInstance(layer.forward_weights.grad, torch.Tensor)
			self.assertEqual(layer.forward_weights.grad.shape, layer.forward_weights.shape)
			self.assertEqual(layer.forward_weights.grad.device.type, layer.device.type)
			self.assertIsInstance(layer.bias_weights.grad, torch.Tensor)
			self.assertEqual(layer.bias_weights.grad.shape, layer.bias_weights.shape)
			self.assertEqual(layer.bias_weights.grad.device.type, layer.device.type)

