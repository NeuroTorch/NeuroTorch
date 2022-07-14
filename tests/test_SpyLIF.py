import unittest

import torch
import numpy as np

from neurotorch.modules import HeavisidePhiApprox, HeavisideSigmoidApprox
from neurotorch.modules.layers import SpyLIFLayer, LearningType


class TestSpyLIFLayer(unittest.TestCase):
	def test_constructor(self):
		layer = SpyLIFLayer(
			input_size=10,
			output_size=5,
			name="test",
			learning_type=LearningType.BPTT,
			dt=0.1,
			device=torch.device('cpu'),
		)
		self.assertEqual(layer.use_recurrent_connection, True)
		self.assertIs(layer.recurrent_weights, None)
		self.assertEqual(int(layer.input_size), 10)
		self.assertEqual(int(layer.output_size), 5)
		self.assertEqual(layer.name, "test")
		self.assertEqual(layer.learning_type, LearningType.BPTT)
		self.assertEqual(layer.dt, 0.1)
		self.assertEqual(layer.device, torch.device('cpu'))

		layer = SpyLIFLayer(
			input_size=20,
			output_size=10,
			name="test",
			learning_type=LearningType.BPTT,
			dt=0.01,
			device=torch.device('cuda'),
		)
		self.assertEqual(layer.use_recurrent_connection, True)
		self.assertIs(layer.recurrent_weights, None)
		self.assertEqual(int(layer.input_size), 20)
		self.assertEqual(int(layer.output_size), 10)
		self.assertEqual(layer.name, "test")
		self.assertEqual(layer.learning_type, LearningType.BPTT)
		self.assertEqual(layer.dt, 0.01)
		self.assertEqual(layer.device, torch.device('cuda'))

	def test_forward(self):
		layer = SpyLIFLayer(
			input_size=5,
			output_size=2,
			name="test",
			learning_type=LearningType.BPTT,
			dt=0.1,
			device=torch.device('cpu'),
		)
		x = torch.randn(1, 5)
		y, (v, I, z) = layer(x)
		self.assertEqual(y.shape, torch.Size([1, 2]))
		self.assertEqual(I.shape, torch.Size([1, 2]))
		self.assertEqual(v.shape, torch.Size([1, 2]))
		self.assertTrue(torch.allclose(y, z))

	def test_kwargs_default_parameter(self):
		layer = SpyLIFLayer(
			input_size=5,
			output_size=2,
			name="test",
			learning_type=LearningType.BPTT,
			dt=1,
			device=torch.device('cpu'),
		)
		layer.build()
		self.assertTrue(np.isclose(layer.kwargs["tau_syn"], 5.0))
		self.assertTrue(np.isclose(layer.kwargs["tau_mem"], 10.0))
		self.assertTrue(np.isclose(layer.kwargs["threshold"], 1.0))
		self.assertTrue(np.isclose(layer.kwargs["gamma"], 100.0), f"{layer.kwargs['gamma'] = } is not equal to 100")
		self.assertEqual(layer.spike_func, HeavisideSigmoidApprox)
		self.assertEqual(layer.use_recurrent_connection, True)
		self.assertTrue(layer.recurrent_weights.shape, torch.Size([2, 2]))
		self.assertTrue(layer.forward_weights.shape, torch.Size([5, 2]))
		self.assertIsInstance(layer.recurrent_weights, torch.nn.Parameter)
		self.assertIsInstance(layer.forward_weights, torch.nn.Parameter)
		self.assertIsInstance(layer.alpha, torch.Tensor)
		self.assertIsInstance(layer.beta, torch.Tensor)
		self.assertIsInstance(layer.gamma, torch.Tensor)
		self.assertIsInstance(layer.threshold, torch.Tensor)

	def test_kwargs_specified_parameter(self):
		layer = SpyLIFLayer(
			input_size=5,
			output_size=2,
			name="test",
			learning_type=LearningType.BPTT,
			dt=1,
			device=torch.device('cpu'),
			tau_syn=1,
			tau_mem=2,
			use_recurrent_connection=False,
		)
		layer.build()
		self.assertTrue(np.isclose(layer.kwargs["tau_syn"], 1))
		self.assertTrue(np.isclose(layer.kwargs["tau_mem"], 2.0))
		self.assertTrue(np.isclose(layer.kwargs["threshold"], 1))
		self.assertTrue(np.isclose(layer.kwargs["gamma"], 100))
		self.assertEqual(layer.spike_func, HeavisideSigmoidApprox)
		self.assertEqual(layer.use_recurrent_connection, False)
		self.assertIs(layer.recurrent_weights, None)
		self.assertTrue(layer.forward_weights.shape, torch.Size([5, 2]))
		self.assertIsInstance(layer.forward_weights, torch.nn.Parameter)
		self.assertIsInstance(layer.alpha, torch.Tensor)
		self.assertIsInstance(layer.beta, torch.Tensor)
		self.assertIsInstance(layer.gamma, torch.Tensor)
		self.assertIsInstance(layer.threshold, torch.Tensor)

	def test_if_grad(self):
		# with recurrent weights
		layer = SpyLIFLayer(
			input_size=5,
			output_size=2,
			name="test",
			learning_type=LearningType.BPTT,
			dt=1,
			device=torch.device('cpu'),
			use_recurrent_connection=True,
		)
		self.assertIs(layer.recurrent_weights, None)
		layer.build()
		self.assertEqual(layer.use_recurrent_connection, True)
		self.assertIsInstance(layer.recurrent_weights, torch.nn.Parameter)
		self.assertIsInstance(layer.forward_weights, torch.nn.Parameter)
		self.assertTrue(layer.forward_weights.requires_grad)
		self.assertTrue(layer.recurrent_weights.requires_grad)

		# without recurrent weights
		layer = SpyLIFLayer(
			input_size=5,
			output_size=2,
			name="test",
			learning_type=LearningType.BPTT,
			dt=1,
			device=torch.device('cpu'),
			use_recurrent_connection=False,
		)
		self.assertIs(layer.recurrent_weights, None)
		layer.build()
		self.assertEqual(layer.use_recurrent_connection, False)
		self.assertIs(layer.recurrent_weights, None)
		self.assertTrue(layer.forward_weights.requires_grad)

		# learning type None
		layer = SpyLIFLayer(
			input_size=5,
			output_size=2,
			name="test",
			learning_type=LearningType.NONE,
			dt=1,
			device=torch.device('cpu'),
		)
		self.assertIs(layer.recurrent_weights, None)
		layer.build()
		self.assertTrue(layer.use_recurrent_connection)
		self.assertFalse(layer.forward_weights.requires_grad)
		self.assertFalse(layer.recurrent_weights.requires_grad)

	def test_device(self):
		"""
		Test if the layer is working on the correct device. The input is being place on the wrong device.
		The device indicated in the __init__ will be used.
		"""
		layer = SpyLIFLayer(input_size=3, output_size=3)
		input_ = torch.rand(1, 3, device="cpu")
		y, (v, I, z) = layer(input_)
		self.assertIsInstance(y, torch.Tensor)
		self.assertEqual(y.shape, torch.Size([1, 3]))
		self.assertEqual(y.device.type, layer.device.type)
		self.assertIsInstance(v, torch.Tensor)
		self.assertEqual(v.shape, torch.Size([1, 3]))
		self.assertEqual(v.device.type, layer.device.type)
		self.assertEqual(I.shape, torch.Size([1, 3]))
		self.assertEqual(I.device.type, layer.device.type)
		self.assertEqual(layer.forward_weights.device.type, layer.device.type)
		self.assertEqual(layer.recurrent_weights.device.type, layer.device.type)

		layer = SpyLIFLayer(input_size=3, output_size=3, device=torch.device(type="cuda", index=0))
		input_ = torch.rand(1, 3, device="cpu")
		y, (v, I, z) = layer(input_)
		self.assertIsInstance(y, torch.Tensor)
		self.assertEqual(y.shape, torch.Size([1, 3]))
		self.assertEqual(y.device.type, layer.device.type)
		self.assertIsInstance(v, torch.Tensor)
		self.assertEqual(v.shape, torch.Size([1, 3]))
		self.assertEqual(v.device.type, layer.device.type)
		self.assertEqual(I.shape, torch.Size([1, 3]))
		self.assertEqual(I.device.type, layer.device.type)
		self.assertEqual(layer.forward_weights.device.type, layer.device.type)
		self.assertEqual(layer.recurrent_weights.device.type, layer.device.type)

		layer = SpyLIFLayer(input_size=3, output_size=3, device=torch.device("cpu"))
		input_ = torch.rand(1, 3, device=torch.device(type="cuda", index=0))
		y, (v, I, z) = layer(input_)
		self.assertIsInstance(y, torch.Tensor)
		self.assertEqual(y.shape, torch.Size([1, 3]))
		self.assertEqual(y.device.type, layer.device.type)
		self.assertIsInstance(v, torch.Tensor)
		self.assertEqual(v.shape, torch.Size([1, 3]))
		self.assertEqual(v.device.type, layer.device.type)
		self.assertEqual(I.shape, torch.Size([1, 3]))
		self.assertEqual(I.device.type, layer.device.type)
		self.assertEqual(layer.forward_weights.device.type, layer.device.type)
		self.assertEqual(layer.recurrent_weights.device.type, layer.device.type)

	def test_backward(self):
		"""
		Test if the backward method works correctly
		"""
		layer = SpyLIFLayer(input_size=3, output_size=3, device=torch.device("cpu"))
		input_ = torch.rand(1, 3)
		output = layer(input_)[0]
		output.mean().backward()
		self.assertIsInstance(layer.forward_weights.grad, torch.Tensor)
		self.assertEqual(layer.forward_weights.grad.shape, layer.forward_weights.shape)
		self.assertEqual(layer.forward_weights.grad.device, layer.device)
		self.assertIsInstance(layer.recurrent_weights.grad, torch.Tensor)
		self.assertEqual(layer.recurrent_weights.grad.shape, layer.recurrent_weights.shape)
		self.assertEqual(layer.recurrent_weights.grad.device, layer.device)

		layer = SpyLIFLayer(
			input_size=3, output_size=3, device=torch.device(type="cuda", index=0), use_recurrent_connection=False
		)
		input_ = torch.rand(1, 3)
		output = layer(input_)[0]
		output.mean().backward()
		self.assertIsInstance(layer.forward_weights.grad, torch.Tensor)
		self.assertEqual(layer.forward_weights.grad.shape, layer.forward_weights.shape)
		self.assertEqual(layer.forward_weights.grad.device, layer.device)
		self.assertIs(layer.recurrent_weights, None)
