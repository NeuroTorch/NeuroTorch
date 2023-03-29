import unittest
import warnings

import numpy as np
import torch

from neurotorch.modules.layers import WilsonCowanLayer


class TestWilsonCowanLayer(unittest.TestCase):

	def test_kwargs_default_parameter(self):
		"""
		Test if the parameter are well initialize in the kwargs
		"""
		# If unspecified
		layer = WilsonCowanLayer()
		self.assertEqual(layer.std_weight, 1.0)
		self.assertEqual(layer.tau, 1.0)
		self.assertEqual(layer.mu, 0.0)
		self.assertEqual(layer.learn_mu, False)
		self.assertEqual(layer.mean_mu, 2.0)
		self.assertEqual(layer.std_mu, 0.0)
		self.assertEqual(layer.dt, 0.001)
		self.assertEqual(layer.r, 0.0)
		self.assertEqual(layer.learn_r, False)
		self.assertEqual(layer.mean_r, 2.0)
		self.assertEqual(layer.std_r, 0.0)

		# If specified
		layer = WilsonCowanLayer(std_weight=10.0, mu=-2.0, tau=3.0, learn_mu=True, mean_mu=4.0, std_mu=5.0,
		                         dt=0.1, r=6.5, learn_r=True, mean_r=7.0, std_r=8.0)
		self.assertEqual(layer.std_weight, 10.0)
		self.assertEqual(layer.tau, 3.0)
		self.assertEqual(layer.mu, -2.0)
		self.assertEqual(layer.learn_mu, True)
		self.assertEqual(layer.mean_mu, 4.0)
		self.assertEqual(layer.std_mu, 5.0)
		self.assertEqual(layer.dt, 0.1)
		self.assertEqual(layer.r, 6.5)
		self.assertEqual(layer.learn_r, True)
		self.assertEqual(layer.mean_r, 7.0)
		self.assertEqual(layer.std_r, 8.0)

	def test_instance(self):
		"""
		Test if the variables are initialized with the correct instance
		"""
		layer = WilsonCowanLayer()
		self.assertIsInstance(layer.std_weight, float)
		self.assertIsInstance(layer.mu, torch.Tensor)
		self.assertIsInstance(layer.mean_mu, float)
		self.assertIsInstance(layer.std_mu, float)
		self.assertIsInstance(layer.tau, torch.Tensor)
		self.assertIsInstance(layer.learn_mu, bool)
		self.assertIs(layer.input_size, None)
		self.assertIs(layer.output_size, None)
		self.assertIsInstance(layer.dt, float)
		self.assertIsInstance(layer.r, torch.Tensor)
		self.assertIsInstance(layer.learn_r, bool)
		self.assertIsInstance(layer.mean_r, float)
		self.assertIsInstance(layer.std_r, float)

		layer = WilsonCowanLayer(learn_mu=True, learn_r=True)
		self.assertIsInstance(layer.mu, torch.Tensor)
		self.assertIsInstance(layer.r, torch.Tensor)

	def test_if_grad(self):
		"""
		Test if the gradient is computed for the desired parameters
		"""
		layer = WilsonCowanLayer(input_size=10, output_size=10, learn_mu=False, learn_r=False)
		layer.build()
		self.assertIs(layer.mu.requires_grad, False)
		self.assertIs(layer.forward_weights.requires_grad, True)
		self.assertIs(layer.r.requires_grad, False)

		layer = WilsonCowanLayer(input_size=10, output_size=10, learn_mu=True, learn_r=True)
		layer.build()
		self.assertIs(layer.mu.requires_grad, True)
		self.assertIs(layer.forward_weights.requires_grad, True)
		self.assertIs(layer.r.requires_grad, True)

	def test_mu_r_dtype(self):
		"""
		Test if mu and r have the correct dtype even if the user gives a wrong dtype
		"""
		layer = WilsonCowanLayer(input_size=10, output_size=10)
		layer.build()
		self.assertEqual(layer.mu.dtype, torch.float32)
		self.assertEqual(layer.r.dtype, torch.float32)

		mu = np.random.rand(10).astype(np.float64)
		r = np.random.rand(10).astype(np.float64)
		layer = WilsonCowanLayer(input_size=10, output_size=10, mu=mu, r=r)
		self.assertEqual(layer.mu.dtype, torch.float32)
		self.assertEqual(layer.r.dtype, torch.float32)

		mu = torch.tensor(2.0, dtype=torch.float64)
		r = torch.tensor(2.0, dtype=torch.float64)
		layer = WilsonCowanLayer(input_size=10, output_size=10, mu=mu, r=r)
		self.assertEqual(layer.mu.dtype, torch.float32)
		self.assertEqual(layer.r.dtype, torch.float32)

	def test_device(self):
		"""
		Test if the layer is working on the correct device. The input is being place on the wrong device.
		The device indicated in the __init__ will be used.
		"""
		mu = torch.rand(1, 3, device='cpu')
		r = torch.rand(1, 3, device='cpu')
		layer = WilsonCowanLayer(input_size=3, output_size=3, mu=mu, r=r)
		input_ = torch.rand(1, 3, device="cpu")
		output = layer(input_)
		self.assertIsInstance(output[0], torch.Tensor)
		self.assertEqual(output[0].shape, (1, 3))
		self.assertEqual(output[0].device.type, layer.device.type)
		self.assertEqual(layer.mu.device.type, layer.device.type)
		self.assertEqual(layer.r.device.type, layer.device.type)

		if torch.cuda.is_available():
			mu = torch.rand(1, 3, device='cpu')
			r = torch.rand(1, 3, device='cpu')
			layer = WilsonCowanLayer(input_size=3, output_size=3, device=torch.device(type="cuda", index=0), mu=mu, r=r)
			input_ = torch.rand(1, 3, device="cpu")
			output = layer(input_)
			self.assertIsInstance(output[0], torch.Tensor)
			self.assertEqual(output[0].shape, (1, 3))
			self.assertEqual(output[0].device, layer.device)
			self.assertEqual(layer.mu.device, layer.device)
			self.assertEqual(layer.r.device, layer.device)

			mu = torch.rand(1, 3, device=torch.device(type="cuda", index=0))
			r = torch.rand(1, 3, device=torch.device(type="cuda", index=0))
			layer = WilsonCowanLayer(input_size=3, output_size=3, device=torch.device("cpu"), mu=mu, r=r)
			input_ = torch.rand(1, 3, device=torch.device(type="cuda", index=0))
			output = layer(input_)
			self.assertIsInstance(output[0], torch.Tensor)
			self.assertEqual(output[0].shape, (1, 3))
			self.assertEqual(output[0].device, layer.device)
			self.assertEqual(layer.mu.device, layer.device)
			self.assertEqual(layer.r.device, layer.device)
		else:
			warnings.warn(
				"No CUDA available. Skipping test_device. Please consider running the tests on a machine with CUDA.",
				UserWarning,
			)

	def test_forward(self):
		"""
		Test if the forward method works correctly by returning a coherent output
		"""
		mu = torch.rand(1, 3)
		r = torch.rand(1, 3)
		layer = WilsonCowanLayer(input_size=3, output_size=3, mu=mu, r=r)
		input_ = torch.rand(1, 3)
		output = layer(input_)
		self.assertIsInstance(output[0], torch.Tensor)
		self.assertEqual(output[0].shape, (1, 3))

	def test_intialize_weights(self):
		"""
		Test if the random weights is initialize with the correct size, mean and STD
		"""
		# If mu is a parameter
		layer = WilsonCowanLayer(
			input_size=500, output_size=500, std_weight=6.0, learn_mu=True, mean_mu=2.0,
			std_mu=1.0, learn_r=True, mean_r=5.0, std_r=3.0
		)
		layer.build()
		self.assertEqual(torch.round(layer.forward_weights.detach().mean()), 0.0)
		self.assertEqual(torch.round(layer.forward_weights.detach().std()), 6.0)
		self.assertEqual(torch.round(layer.mu.detach().mean()), 2.0)
		self.assertEqual(torch.round(layer.mu.detach().std()), 1.0)
		self.assertEqual(torch.round(layer.r.detach().mean()), 5.0)
		self.assertEqual(torch.round(layer.r.detach().std()), 3.0)

		# if mu and r are not a parameter
		mu = torch.rand(500, 1)
		r = torch.rand(500, 1)
		layer = WilsonCowanLayer(input_size=500, output_size=500, std_weight=6.0, learn_mu=False, mu=mu, r=r)
		layer.build()
		self.assertEqual(torch.round(layer.forward_weights.detach().mean()), 0.0)
		self.assertEqual(torch.round(layer.forward_weights.detach().std()), 6.0)
		self.assertEqual(layer.mu.all(), mu.all())
		self.assertEqual(layer.r.all(), r.all())

	def test_backward(self):
		"""
		Test if the backward method works correctly
		"""
		layer = WilsonCowanLayer(input_size=3, output_size=3, device=torch.device("cpu"))
		input_ = torch.rand(1, 3)
		output = layer(input_)[0]
		output.mean().backward()
		self.assertIsInstance(layer.forward_weights.grad, torch.Tensor)
		self.assertEqual(layer.forward_weights.grad.shape, layer.forward_weights.shape)
		self.assertEqual(layer.forward_weights.grad.device, layer.device)
		self.assertEqual(layer.mu.grad, None)
		self.assertEqual(layer.r.grad, None)

		if torch.cuda.is_available():
			mu = torch.rand(1, 3)
			r = torch.rand(1, 3)
			layer = WilsonCowanLayer(
				input_size=3, output_size=3, device=torch.device(type="cuda", index=0), mu=mu, r=r,
				learn_mu=True, learn_r=True, learn_tau=True
			)
			input_ = torch.rand(1, 3)
			output = layer(input_)[0]
			output.mean().backward()
			self.assertIsInstance(layer.forward_weights.grad, torch.Tensor)
			self.assertEqual(layer.forward_weights.grad.shape, layer.forward_weights.shape)
			self.assertEqual(layer.forward_weights.grad.device, layer.device)
			self.assertIsInstance(layer.mu.grad, torch.Tensor)
			self.assertIsInstance(layer.r_sqrt.grad, torch.Tensor)
			self.assertIsInstance(layer.tau_sqrt.grad, torch.Tensor)
		else:
			warnings.warn(
				"No CUDA available. Skipping test_backward. Please consider running the tests on a machine with CUDA.",
				UserWarning,
			)

	def test_output_result(self):
		"""
		Test if the output result match the true data
		"""
		layer = WilsonCowanLayer(input_size=60, output_size=60)
		input_ = torch.rand(1, 60, device=layer._device)
		output = layer(input_)[0]
		ratio_dt_tau = layer.dt / layer.tau
		transition_rate = (1 - layer.r * input_)
		sigmoid = (torch.sigmoid(torch.matmul(input_, layer.forward_weights) - layer.mu))
		true_output = input_ * (1 - ratio_dt_tau) + transition_rate * sigmoid * ratio_dt_tau
		self.assertEqual(output.all(), true_output.all())

		# If mu and r have a default value
		mu = torch.rand(1, 60, device="cpu")
		r = torch.rand(1, 60, device="cpu")
		layer = WilsonCowanLayer(input_size=60, output_size=60, mu=mu)
		input_ = torch.rand(1, 60, device=layer._device)
		output = layer(input_)[0]
		ratio_dt_tau = layer.dt / layer.tau
		transition_rate = (1 - layer.r * input_)
		sigmoid = (torch.sigmoid(torch.matmul(input_, layer.forward_weights) - layer.mu))
		true_output = input_ * (1 - ratio_dt_tau) + transition_rate * sigmoid * ratio_dt_tau
		self.assertEqual(output.all(), true_output.all())

		# If input_ has multiple time steps
		mu = torch.rand(1, 60, device="cpu")
		r = torch.rand(1, 60, device="cpu")
		layer = WilsonCowanLayer(input_size=60, output_size=60, mu=mu)
		input_ = torch.rand(10, 60, device=layer._device)
		output = layer(input_)[0]
		ratio_dt_tau = layer.dt / layer.tau
		transition_rate = (1 - layer.r * input_)
		sigmoid = (torch.sigmoid(torch.matmul(input_, layer.forward_weights) - layer.mu))
		true_output = input_ * (1 - ratio_dt_tau) + transition_rate * sigmoid * ratio_dt_tau
		self.assertEqual(output.all(), true_output.all())

	def test_get_sign_parameters_force_dale_law_true(self):
		"""
		Test if the force_dale_law is working correctly
		"""
		layer = WilsonCowanLayer(
			input_size=60, output_size=60, force_dale_law=True, use_recurrent_connection=False
		).build()

		for param in layer.get_sign_parameters():
			self.assertIsInstance(param, torch.nn.Parameter)

	def test_get_sign_parameters_force_dale_law_true_rec(self):
		"""
		Test if the force_dale_law is working correctly
		"""
		layer = WilsonCowanLayer(
			input_size=60, output_size=60, force_dale_law=True, use_recurrent_connection=True
		).build()

		for param in layer.get_sign_parameters():
			self.assertIsInstance(param, torch.nn.Parameter)

	def test_get_weights_parameters_force_dale_law_true(self):
		"""
		Test if the force_dale_law is working correctly
		"""
		layer = WilsonCowanLayer(
			input_size=60, output_size=60, force_dale_law=True, use_recurrent_connection=False
		).build()

		for param in layer.get_weights_parameters():
			self.assertIsInstance(param, torch.nn.Parameter)

	def test_get_weights_parameters_force_dale_law_true_rec(self):
		"""
		Test if the force_dale_law is working correctly
		"""
		layer = WilsonCowanLayer(
			input_size=60, output_size=60, force_dale_law=True, use_recurrent_connection=True
		).build()

		for param in layer.get_weights_parameters():
			self.assertIsInstance(param, torch.nn.Parameter)


if __name__ == '__main__':
	unittest.main()
