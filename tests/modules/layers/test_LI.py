import unittest
import warnings

import torch
import numpy as np

from neurotorch.modules.layers import LILayer


class TestLILayer(unittest.TestCase):
    def test_constructor(self):
        layer = LILayer(
            input_size=10,
            output_size=5,
            name="test",
            dt=0.1,
            device=torch.device('cpu'),
        )
        self.assertEqual(layer.use_recurrent_connection, False)
        self.assertIs(layer.recurrent_weights, None)
        self.assertEqual(int(layer.input_size), 10)
        self.assertEqual(int(layer.output_size), 5)
        self.assertEqual(layer.name, "test")
        self.assertEqual(layer.dt, 0.1)
        self.assertEqual(layer.device, torch.device('cpu'))

        if torch.cuda.is_available():
            layer = LILayer(
                input_size=20,
                output_size=10,
                name="test",
                dt=0.01,
                device=torch.device('cuda'),
            )
            self.assertEqual(layer.use_recurrent_connection, False)
            self.assertIs(layer.recurrent_weights, None)
            self.assertEqual(int(layer.input_size), 20)
            self.assertEqual(int(layer.output_size), 10)
            self.assertEqual(layer.name, "test")
            self.assertEqual(layer.dt, 0.01)
            self.assertEqual(layer.device, torch.device('cuda'))
        else:
            warnings.warn(
                "No CUDA available. Skipping test_constructor. Please consider running the tests on a machine with CUDA.",
                UserWarning,
            )

    def test_forward(self):
        layer = LILayer(
            input_size=5,
            output_size=2,
            name="test",
            dt=0.1,
            device=torch.device('cpu'),
        )
        x = torch.randn(1, 5)
        y, (hh, ) = layer(x)
        self.assertEqual(y.shape, torch.Size([1, 2]))
        self.assertEqual(hh.shape, torch.Size([1, 2]))
        self.assertTrue(torch.allclose(y, hh))

    def test_kwargs_default_parameter(self):
        layer = LILayer(
            input_size=5,
            output_size=2,
            name="test",
            dt=1,
            device=torch.device('cpu'),
        )
        self.assertIs(layer.bias_weights, None)
        layer.build()
        self.assertEqual(layer.use_recurrent_connection, False)
        self.assertIs(layer.recurrent_weights, None)
        self.assertEqual(layer.bias_weights.shape, torch.Size([2]))
        self.assertTrue(np.isclose(layer.kwargs["tau_out"], 10.0))

    def test_kwargs_specified_parameter(self):
        layer = LILayer(
            input_size=5,
            output_size=2,
            name="test",
            dt=1,
            device=torch.device('cpu'),
            tau_out=1,
        )
        self.assertIs(layer.bias_weights, None)
        layer.build()
        self.assertEqual(layer.use_recurrent_connection, False)
        self.assertIs(layer.recurrent_weights, None)
        self.assertEqual(layer.bias_weights.shape, torch.Size([2]))
        self.assertTrue(np.isclose(layer.kwargs["tau_out"], 1))

        layer = LILayer(
            input_size=5,
            output_size=2,
            name="test",
            dt=1,
            device=torch.device('cpu'),
            tau_out=1,
            use_bias=False,
        )
        self.assertIs(layer.bias_weights, None)
        layer.build()
        self.assertEqual(layer.use_recurrent_connection, False)
        self.assertIs(layer.recurrent_weights, None)
        self.assertEqual(layer.bias_weights.shape, torch.Size([2]))
        self.assertTrue(torch.allclose(layer.bias_weights, torch.zeros([2])))
        self.assertTrue(np.isclose(layer.kwargs["tau_out"], 1))

    def test_if_grad(self):
        # with bias
        layer = LILayer(
            input_size=5,
            output_size=2,
            name="test",
            dt=1,
            device=torch.device('cpu'),
        )
        self.assertIs(layer.bias_weights, None)
        self.assertIs(layer.recurrent_weights, None)
        layer.build()
        self.assertEqual(layer.use_recurrent_connection, False)
        self.assertIs(layer.recurrent_weights, None)
        self.assertTrue(layer.forward_weights.requires_grad)
        self.assertTrue(layer.bias_weights.requires_grad)

        # without bias
        layer = LILayer(
            input_size=5,
            output_size=2,
            name="test",
            dt=1,
            device=torch.device('cpu'),
            use_bias=False,
        )
        self.assertIs(layer.bias_weights, None)
        self.assertIs(layer.recurrent_weights, None)
        layer.build()
        self.assertEqual(layer.use_recurrent_connection, False)
        self.assertIs(layer.recurrent_weights, None)
        self.assertTrue(layer.forward_weights.requires_grad)
        self.assertFalse(layer.bias_weights.requires_grad)

        # frozen
        layer = LILayer(
            input_size=5,
            output_size=2,
            name="test",
            dt=1,
            freeze_weights=True,
            device=torch.device('cpu'),
        )
        self.assertIs(layer.bias_weights, None)
        self.assertIs(layer.recurrent_weights, None)
        layer.build()
        self.assertEqual(layer.use_recurrent_connection, False)
        self.assertIs(layer.recurrent_weights, None)
        self.assertFalse(layer.forward_weights.requires_grad)
        self.assertFalse(layer.bias_weights.requires_grad)

    def test_device(self):
        """
        Test if the layer is working on the correct device. The input is being place on the wrong device.
        The device indicated in the __init__ will be used.
        """
        layer = LILayer(input_size=3, output_size=3)
        input_ = torch.rand(1, 3, device="cpu")
        y, (hh, ) = layer(input_)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.shape, torch.Size([1, 3]))
        self.assertEqual(y.device.type, layer.device.type)
        self.assertIsInstance(hh, torch.Tensor)
        self.assertEqual(hh.shape, torch.Size([1, 3]))
        self.assertEqual(hh.device.type, layer.device.type)
        self.assertEqual(layer.forward_weights.device.type, layer.device.type)
        self.assertEqual(layer.bias_weights.device.type, layer.device.type)

        if torch.cuda.is_available():
            layer = LILayer(input_size=3, output_size=3, device=torch.device(type="cuda", index=0))
            input_ = torch.rand(1, 3, device="cpu")
            y, (hh,) = layer(input_)
            self.assertIsInstance(y, torch.Tensor)
            self.assertEqual(y.shape, torch.Size([1, 3]))
            self.assertEqual(y.device.type, layer.device.type)
            self.assertIsInstance(hh, torch.Tensor)
            self.assertEqual(hh.shape, torch.Size([1, 3]))
            self.assertEqual(hh.device.type, layer.device.type)
            self.assertEqual(layer.forward_weights.device.type, layer.device.type)
            self.assertEqual(layer.bias_weights.device.type, layer.device.type)

            layer = LILayer(input_size=3, output_size=3, device=torch.device("cpu"))
            input_ = torch.rand(1, 3, device=torch.device(type="cuda", index=0))
            y, (hh,) = layer(input_)
            self.assertIsInstance(y, torch.Tensor)
            self.assertEqual(y.shape, torch.Size([1, 3]))
            self.assertEqual(y.device.type, layer.device.type)
            self.assertIsInstance(hh, torch.Tensor)
            self.assertEqual(hh.shape, torch.Size([1, 3]))
            self.assertEqual(hh.device.type, layer.device.type)
            self.assertEqual(layer.forward_weights.device.type, layer.device.type)
            self.assertEqual(layer.bias_weights.device.type, layer.device.type)
        else:
            warnings.warn(
                "No CUDA available. Skipping test_device. Please consider running the tests on a machine with CUDA.",
                UserWarning,
            )

    def test_backward(self):
        """
        Test if the backward method works correctly
        """
        layer = LILayer(input_size=3, output_size=3, device=torch.device("cpu"))
        input_ = torch.rand(1, 3)
        output = layer(input_)[0]
        output.mean().backward()
        self.assertIsInstance(layer.forward_weights.grad, torch.Tensor)
        self.assertEqual(layer.forward_weights.grad.shape, layer.forward_weights.shape)
        self.assertEqual(layer.forward_weights.grad.device, layer.device)
        self.assertIsInstance(layer.bias_weights.grad, torch.Tensor)
        self.assertEqual(layer.bias_weights.grad.shape, layer.bias_weights.shape)
        self.assertEqual(layer.bias_weights.grad.device, layer.device)

        if torch.cuda.is_available():
            layer = LILayer(input_size=3, output_size=3, device=torch.device(type="cuda", index=0), use_bias=False)
            input_ = torch.rand(1, 3)
            output = layer(input_)[0]
            output.mean().backward()
            self.assertIsInstance(layer.forward_weights.grad, torch.Tensor)
            self.assertEqual(layer.forward_weights.grad.shape, layer.forward_weights.shape)
            self.assertEqual(layer.forward_weights.grad.device, layer.device)
            self.assertIs(layer.bias_weights.grad, None)
        else:
            warnings.warn(
                "No CUDA available. Skipping test_backward. Please consider running the tests on a machine with CUDA.",
                UserWarning,
            )
