import unittest
import warnings
from typing import Iterable
from functools import partial

import numpy as np
import torch
from torchvision.transforms import Compose

from neurotorch.modules import ALIFLayer, LIFLayer, LILayer
from neurotorch.modules import SequentialRNN, BaseLayer
from neurotorch import Dimension, DimensionProperty, Linear
from neurotorch.utils import ravel_compose_transforms


class TestSequentialRNN(unittest.TestCase):
    def test_init_base(self):
        model_name = "input"
        model = SequentialRNN(
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
        model = SequentialRNN(
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
        model = SequentialRNN(
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
        model = SequentialRNN(
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
        model = SequentialRNN(
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
        model = SequentialRNN(
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
        model = SequentialRNN(
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
        model = SequentialRNN(
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
        model = SequentialRNN(
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
        model = SequentialRNN(
            layers=[
                BaseLayer(Dimension(10, DimensionProperty.NONE), 2, name="input"),
            ],
            device=torch.device("cpu"),
        )
        model.build()
        self.assertEqual(model.device, torch.device("cpu"))
        for layer in model.get_all_layers():
            self.assertEqual(layer.device, torch.device("cpu"))

        if torch.cuda.is_available():
            # Test that the model is initialized with the specified device if it is specified in the layer
            model = SequentialRNN(
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
            model = SequentialRNN(
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
            model = SequentialRNN(
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
        else:
            warnings.warn(
                "No CUDA available. Skipping test_init_device_specified. Please consider running the tests on a machine "
                "with CUDA.",
                UserWarning,
            )

    def test_format_hidden_outputs_traces(self):
        data = torch.ones((32, 2))
        time_steps = 10
        hh_states = {
            '0': [(0 * data, 1 * data, 2 * data) for _ in range(time_steps)]
        }
        hh_states_transposed = {
            '0': tuple([torch.stack(e, dim=1) for e in list(zip(*hh_states['0']))])
        }
        hh_pred = SequentialRNN._format_hidden_outputs_traces(hh_states)
        self.assertTrue(all(torch.allclose(x, y) for x, y in zip(hh_states_transposed['0'], hh_pred['0'])))

        hh_states = {
            '0': [(data, ) for _ in range(time_steps)]
        }
        hh_states_transposed = {
            '0': torch.stack([data for _ in range(time_steps)], dim=1)
        }
        hh_pred = SequentialRNN._format_hidden_outputs_traces(hh_states)
        self.assertTrue(all(torch.allclose(x, y) for x, y in zip(hh_states_transposed['0'], hh_pred['0'])))

        hh_states = {
            '0': [data for _ in range(time_steps)]
        }
        hh_states_transposed = {
            '0': torch.stack([data for _ in range(time_steps)], dim=1)
        }
        hh_pred = SequentialRNN._format_hidden_outputs_traces(hh_states)
        self.assertTrue(torch.allclose(hh_states_transposed['0'], hh_pred['0']))

        hh_states = {
            '0': [(None, ) for _ in range(time_steps)]
        }
        hh_states_transposed = {
            '0': [None for _ in range(time_steps)]
        }
        hh_pred = SequentialRNN._format_hidden_outputs_traces(hh_states)
        self.assertEqual(hh_states_transposed['0'],  hh_pred['0'])

        hh_states = {
            '0': [None for _ in range(time_steps)]
        }
        hh_states_transposed = {
            '0': [None for _ in range(time_steps)]
        }
        hh_pred = SequentialRNN._format_hidden_outputs_traces(hh_states)
        self.assertEqual(hh_states_transposed['0'], hh_pred['0'])

    def test_init_transforms(self):
        # Test that the model is initialized with the default transforms with one layer
        model = SequentialRNN(
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
        model = SequentialRNN(
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

        class _DummyTransform(torch.nn.Module):
            def forward(self, x):
                return x

        trans = _DummyTransform()
        # Test that the model is initialized with the specified transforms
        model = SequentialRNN(
            layers=[
                BaseLayer(Dimension(10, DimensionProperty.NONE), 2),
                BaseLayer(Dimension(10, DimensionProperty.NONE), 2),
            ],
            input_transform=[trans]
        )
        model.build()
        self.assertEqual(len(model.input_transform), 1)
        for key, value in model.input_transform.items():
            self.assertIn(
                key, model.input_layers.keys(),
                f"{key} is not in the output layers {model.input_layers.keys()}"
            )
            self.assertIn(trans, ravel_compose_transforms(value))

    def test_get_prediction_trace_forward(self):
        model = SequentialRNN(
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

        self.assertEqual(model.foresight_time_steps, 100)

        x = torch.randn(1, 100, 10)
        y = model.get_prediction_trace(x)
        self.assertTrue(y.requires_grad)
        self.assertEqual(y.shape, torch.Size([1, model.foresight_time_steps, 10]))

        x = torch.randn(1, 1000, 10)
        y = model.get_prediction_trace(x)
        self.assertTrue(y.requires_grad)
        self.assertEqual(y.shape, torch.Size([1, model.foresight_time_steps, 10]))

        x = torch.randn(1, 1000, 10)
        y = model.get_prediction_trace(x, foresight_time_steps=50)
        self.assertTrue(y.requires_grad)
        self.assertEqual(y.shape, torch.Size([1, model.foresight_time_steps, 10]))

        x = torch.randn(1, 1000, 10)
        y = model.get_prediction_trace(x, foresight_time_steps=200)
        self.assertTrue(y.requires_grad)
        self.assertEqual(y.shape, torch.Size([1, model.foresight_time_steps, 10]))

        x = torch.randn(1, 1000, 10)
        y = model.get_prediction_trace(x, foresight_time_steps=None)
        self.assertTrue(y.requires_grad)
        self.assertEqual(y.shape, torch.Size([1, model.foresight_time_steps, 10]))

        model.out_memory_size = 10
        x = torch.randn(1, 1000, 10)
        y = model.get_prediction_trace(x, foresight_time_steps=200)
        self.assertTrue(y.requires_grad)
        self.assertEqual(y.shape, torch.Size([1, model.out_memory_size, 10]))

        model.out_memory_size = 10
        x = torch.randn(1, 1000, 10)
        y = model.get_prediction_trace(x)
        self.assertTrue(y.requires_grad)
        self.assertEqual(y.shape, torch.Size([1, model.out_memory_size, 10]))

        model.out_memory_size = 10
        x = torch.randn(1, 1000, 10)
        y = model.get_prediction_trace(x, foresight_time_steps=50)
        self.assertTrue(y.requires_grad)
        self.assertEqual(y.shape, torch.Size([1, min(model.out_memory_size, 50), 10]))

    def test_if_grad(self):
        model = SequentialRNN(
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
        model = SequentialRNN(
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
        model = SequentialRNN(
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
        model = SequentialRNN(
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
        y, hh = model.get_raw_prediction(x)
        for key, value in y.items():
            self.assertTrue(value.requires_grad)
            self.assertEqual(value.shape, torch.Size([1, x.shape[-2], 2]))
            value.mean().backward()

        for layer in model.get_all_layers():
            self.assertIsInstance(layer.forward_weights.grad, torch.Tensor)
            self.assertEqual(layer.forward_weights.grad.shape, layer.forward_weights.shape)
            self.assertEqual(layer.forward_weights.grad.device.type, layer.device.type)
            self.assertIsInstance(layer.bias_weights.grad, torch.Tensor)
            self.assertEqual(layer.bias_weights.grad.shape, layer.bias_weights.shape)
            self.assertEqual(layer.bias_weights.grad.device.type, layer.device.type)

    def test_get_prediction_proba_backward(self):
        model = SequentialRNN(
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
        model = SequentialRNN(
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

    def test_get_and_reset_regularization_loss(self):
        layers = [
            BaseLayer(10, 10) for _ in range(3)
        ]

        def _update(layer_self, x):
            layer_self._regularization_loss = torch.tensor(x)
            return layer_self._regularization_loss

        for layer in layers:
            layer.update_regularization_loss = partial(_update, layer)

        model = SequentialRNN(
            layers=layers
        )
        model.build()
        self.assertTrue(torch.isclose(model.get_and_reset_regularization_loss(), torch.tensor(0.0)))
        for layer in layers:
            layer.update_regularization_loss(0.1)
        self.assertTrue(torch.isclose(model.get_and_reset_regularization_loss(), torch.tensor(0.1)*len(layers)))
        for layer in layers:
            self.assertTrue(torch.isclose(layer.get_regularization_loss(), torch.tensor(0.0)))

    def test_output_shape_with_out_memory_size(self):
        model = SequentialRNN(
            layers=[
                LILayer(10, 10),
            ],
        ).build()
        x = torch.randn(1, 100, 10)
        y = model.get_prediction_trace(x)
        self.assertEqual(y.shape, torch.Size((1, 100, 10)))

        model = SequentialRNN(
            layers=[
                LILayer(10, 10),
            ],
            out_memory_size=5,
        ).build()
        x = torch.randn(1, 100, 10)
        y = model.get_prediction_trace(x)
        self.assertEqual(y.shape, torch.Size((1, 5, 10)))

        model = SequentialRNN(
            layers=[
                LILayer(10, 10),
            ],
        ).build()
        x = torch.randn(1, 100, 10)
        y = model.get_prediction_trace(x, foresight_time_steps=20)
        self.assertEqual(y.shape, torch.Size((1, 119, 10)), msg=f"Shape of y is {y.shape} and should be (1, 5, 10)")

        model = SequentialRNN(
            layers=[
                LILayer(10, 10),
            ],
        ).build()
        x = torch.randn(1, 100, 10)
        y = model.get_prediction_trace(x, foresight_time_steps=200, trunc_time_steps=20)
        self.assertEqual(y.shape, torch.Size((1, 20, 10)), msg=f"Shape of y is {y.shape} and should be (1, 5, 10)")

        model = SequentialRNN(
            layers=[
                LILayer(10, 10),
            ],
            out_memory_size=5,
        ).build()
        x = torch.randn(1, 100, 10)
        y = model.get_prediction_trace(x, foresight_time_steps=20, trunc_time_steps=200)
        self.assertEqual(y.shape, torch.Size((1, 5, 10)), msg=f"Shape of y is {y.shape} and should be (1, 5, 10)")

        model = SequentialRNN(
            layers=[
                LILayer(10, 10),
            ],
            out_memory_size=5,
            foresight_time_steps=20
        ).build()
        x = torch.randn(1, 100, 10)
        y = model.get_prediction_trace(x, trunc_time_steps=200)
        self.assertEqual(y.shape, torch.Size((1, 5, 10)), msg=f"Shape of y is {y.shape} and should be (1, 5, 10)")

    def test_output_shape_with_hh_memory_size(self):
        model = SequentialRNN(
            layers=[
                LILayer(10, 10, name='layer'),
            ],
        ).build()
        x = torch.randn(2, 100, 10)
        y, hh = model.get_prediction_trace(x, return_hidden_states=True)
        if isinstance(hh, dict):
            hh = hh["layer"]
        if isinstance(hh, tuple):
            hh = hh[0]
        self.assertEqual(hh.shape[1], 100, msg=f"Shape of hh is {hh.shape} and should be (1, 100, 10)")

        model = SequentialRNN(
            layers=[
                LILayer(10, 10, name='layer'),
            ],
            hh_memory_size=5,
        ).build()
        x = torch.randn(1, 100, 10)
        y, hh = model.get_prediction_trace(x, return_hidden_states=True)
        if isinstance(hh, dict):
            hh = hh["layer"]
        if isinstance(hh, tuple):
            hh = hh[0]
        self.assertEqual(hh.shape[1], 5)

        model = SequentialRNN(
            layers=[
                LILayer(10, 10, name='layer'),
            ],
        ).build()
        x = torch.randn(1, 100, 10)
        y, hh = model.get_prediction_trace(x, foresight_time_steps=20, return_hidden_states=True)
        if isinstance(hh, dict):
            hh = hh["layer"]
        if isinstance(hh, tuple):
            hh = hh[0]
        self.assertEqual(hh.shape[1], 119)

        model = SequentialRNN(
            layers=[
                LILayer(10, 10, name='layer'),
            ],
        ).build()
        x = torch.randn(1, 100, 10)
        y, hh = model.get_prediction_trace(x, foresight_time_steps=200, trunc_time_steps=20, return_hidden_states=True)
        if isinstance(hh, dict):
            hh = hh["layer"]
        if isinstance(hh, tuple):
            hh = hh[0]
        self.assertEqual(hh.shape[1], 20)

        model = SequentialRNN(
            layers=[
                LILayer(10, 10, name='layer'),
            ],
            hh_memory_size=5,
        ).build()
        x = torch.randn(1, 100, 10)
        y, hh = model.get_prediction_trace(x, foresight_time_steps=20, trunc_time_steps=200, return_hidden_states=True)
        if isinstance(hh, dict):
            hh = hh["layer"]
        if isinstance(hh, tuple):
            hh = hh[0]
        self.assertEqual(hh.shape[1], 5)

        model = SequentialRNN(
            layers=[
                LILayer(10, 10, name='layer'),
            ],
            hh_memory_size=5,
            foresight_time_steps=20
        ).build()
        x = torch.randn(1, 100, 10)
        y, hh = model.get_prediction_trace(x, trunc_time_steps=200, return_hidden_states=True)
        if isinstance(hh, dict):
            hh = hh["layer"]
        if isinstance(hh, tuple):
            hh = hh[0]
        self.assertEqual(hh.shape[1], 5)

    def test_to(self):
        """
        Test that the to method works as expected.
        :return: None
        """
        model = SequentialRNN(layers=[Linear(10, 10), Linear(10, 10)]).build()
        model.to(torch.device("cpu"))
        self.assertEqual(model.device.type, 'cpu', f"{model.device = }, expected 'cpu'")
        for layer in model.get_layers():
            self.assertEqual(layer.device.type, 'cpu', f"{layer.device = }, expected 'cpu'")
        for m in model.modules():
            if hasattr(m, 'device'):
                self.assertEqual(m.device.type, 'cpu', f"{m.device = }, expected 'cpu'")
        for p in model.parameters():
            self.assertEqual(p.device.type, 'cpu', f"{p.device = }, expected 'cpu'")

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
            self.assertEqual(model.device.type, 'cuda', f"{model.device = }, expected 'cuda'")
            for layer in model.get_layers():
                self.assertEqual(layer.device.type, 'cuda', f"{layer.device = }, expected 'cuda'")
            for m in model.modules():
                if hasattr(m, 'device'):
                    self.assertEqual(m.device.type, 'cuda', f"{m.device = }, expected 'cuda'")
            for p in model.parameters():
                self.assertEqual(p.device.type, 'cuda', f"{p.device = }, expected 'cuda'")
        else:
            warnings.warn(
                "No CUDA available. Skipping test_to. Please consider running the tests on a machine "
                "with CUDA.",
                UserWarning,
            )

    def test_set_device(self):
        model = SequentialRNN(layers=[Linear(10, 10), Linear(10, 10)]).build()
        model.device = torch.device("cpu")
        self.assertEqual(model.device.type, 'cpu', f"{model.device = }, expected 'cpu'")
        for layer in model.get_layers():
            self.assertEqual(layer.device.type, 'cpu', f"{layer.device = }, expected 'cpu'")
        for m in model.modules():
            if hasattr(m, 'device'):
                self.assertEqual(m.device.type, 'cpu', f"{m.device = }, expected 'cpu'")
        for p in model.parameters():
            self.assertEqual(p.device.type, 'cpu', f"{p.device = }, expected 'cpu'")

        if torch.cuda.is_available():
            model.device = torch.device("cuda")
            self.assertEqual(model.device.type, 'cuda', f"{model.device = }, expected 'cuda'")
            for layer in model.get_layers():
                self.assertEqual(layer.device.type, 'cuda', f"{layer.device = }, expected 'cuda'")
            for m in model.modules():
                if hasattr(m, 'device'):
                    self.assertEqual(m.device.type, 'cuda', f"{m.device = }, expected 'cuda'")
            for p in model.parameters():
                self.assertEqual(p.device.type, 'cuda', f"{p.device = }, expected 'cuda'")
        else:
            warnings.warn(
                "No CUDA available. Skipping test_to. Please consider running the tests on a machine "
                "with CUDA.",
                UserWarning,
            )


