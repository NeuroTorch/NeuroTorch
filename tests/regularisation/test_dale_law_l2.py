import random
import unittest

import torch

from neurotorch.regularization.connectome import DaleLawL2


class TestDaleLawL2(unittest.TestCase):
	"""
	Test if the Dale's law is well implemented
	"""

	def test_size_output(self):
		"""
		Assert that the output is a 1D tensor
		"""
		weights_reference = torch.rand(10, 10)
		weights1 = torch.rand(10, 10)
		weights1 = torch.nn.Parameter(weights1)

		weights2 = torch.rand(10, 10)
		weights2 = torch.nn.Parameter(weights2)

		alpha = 0.5
		dale_law = DaleLawL2([weights1, weights2], alpha, weights_reference)
		loss = dale_law()
		self.assertEqual(loss.numel(), 1)

		alpha = 0
		dale_law = DaleLawL2([weights1], alpha, weights_reference)
		loss = dale_law()
		self.assertEqual(loss.numel(), 1)

		alpha = 1
		dale_law = DaleLawL2([weights2], alpha, weights_reference)
		loss = dale_law()
		self.assertEqual(loss.numel(), 1)

	def test_raise_error(self):
		"""
		Assert that the error is raised when needed
		"""
		with self.assertRaises(ValueError):
			weights_reference = torch.rand(10, 10)
			weights = torch.rand(10, 10)
			weights = torch.nn.Parameter(weights)
			alpha = -1
			dale_law = DaleLawL2([weights], alpha, weights_reference)

		with self.assertRaises(ValueError):
			weights_reference = torch.rand(10, 10)
			weights = torch.rand(10, 10)
			weights = torch.nn.Parameter(weights)
			alpha = 2
			dale_law = DaleLawL2([weights], alpha, weights_reference)

		with self.assertRaises(ValueError):
			weights = torch.rand(10, 10)
			weights = torch.nn.Parameter(weights)
			alpha = 0
			dale_law = DaleLawL2([weights], alpha, reference_weights=None)

		try:
			weights = torch.rand(10, 10)
			weights = torch.nn.Parameter(weights)
			dale_law = DaleLawL2([weights], alpha=1, reference_weights=None)
		except ValueError:
			self.fail("DaleLaw should not raise an error when weights_reference is None")

	def test_gradient_is_compute(self):
		"""
		Assert that the gradient is computed
		"""
		weights = torch.rand(10, 10)
		reference_weights = torch.rand(10, 10)
		weights = torch.nn.Parameter(weights)
		alpha = 0.5
		dale_law = DaleLawL2([weights], alpha, reference_weights)
		loss = dale_law()
		loss.backward()
		self.assertTrue(weights.grad is not None)

		weights = torch.rand(10, 10)
		reference_weights = torch.rand(10, 10)
		weights = torch.nn.Parameter(weights)
		alpha = 0
		dale_law = DaleLawL2([weights], alpha, reference_weights)
		loss = dale_law()
		loss.backward()
		self.assertTrue(weights.grad is not None)

		weights = torch.rand(10, 10)
		weights = torch.nn.Parameter(weights, requires_grad=True)
		alpha = 1
		dale_law = DaleLawL2([weights], alpha)
		loss = dale_law()
		loss.backward()
		self.assertTrue(weights.grad is not None)

		weights = torch.rand(10, 10)
		reference_weights = torch.rand(10, 10)
		weights = torch.nn.Parameter(weights, requires_grad=True)
		t = alpha
		dale_law = DaleLawL2([weights], alpha, reference_weights)
		loss = dale_law()
		loss.backward()
		self.assertTrue(weights.grad is not None)

	def test_value_loss_t_is_0(self):
		"""
		Assert that the loss is computed correctly when t is 0. The following scenario are:
		- Weights matrix has the same sign as the reference weights matrix
		- Weights matrix has the opposite sign as the reference weights matrix
		- Weights matrix and reference matrix have random pattern of sign
		:return:
		"""
		weights = torch.rand(10, 10)
		weights = torch.nn.Parameter(weights, requires_grad=True)
		reference_weights = torch.rand(10, 10)
		dale_law = DaleLawL2([weights], alpha=0, reference_weights=reference_weights)
		loss = dale_law()
		self.assertEqual(loss, -torch.trace(weights.detach().T @ torch.sign(reference_weights)))

		weights_init = torch.rand(10, 10)
		weights = torch.nn.Parameter(weights, requires_grad=True)
		reference_weights = torch.rand(10, 10) * -1
		dale_law = DaleLawL2([weights], alpha=0, reference_weights=reference_weights)
		loss_if_bad = dale_law()
		self.assertEqual(
			loss_if_bad,
			-torch.trace(weights.detach().T @ torch.sign(reference_weights))
		)
		weights = torch.nn.Parameter(weights_init * -1, requires_grad=True)
		dale_law = DaleLawL2([weights], alpha=0, reference_weights=reference_weights)
		loss_if_good = dale_law()
		self.assertTrue(loss_if_good < loss_if_bad)

		weights = torch.randn(10, 10)
		weights = torch.nn.Parameter(weights, requires_grad=True)
		reference_weights = torch.randn(10, 10)
		dale_law = DaleLawL2([weights], alpha=0, reference_weights=reference_weights)
		loss = dale_law()
		self.assertEqual(loss, -torch.trace(weights.detach().T @ torch.sign(reference_weights)))

	def test_value_loss_t_1(self):
		"""
		Assert that the loss is computed correctly when t is 1. The following scenario are:
		- Weights matrix has only positive sign
		- Weights matrix has only positive sign but a reference matrix is given
		- Weights matrix is generate randomly with a normal distribution
		:return:
		"""
		weights = torch.rand(10, 10)
		weights = torch.nn.Parameter(weights, requires_grad=True)
		dale_law = DaleLawL2([weights], alpha=1, reference_weights=None)
		loss = dale_law()
		self.assertEqual(torch.round(loss.detach()), torch.round(torch.norm(weights.detach(), p="fro") ** 2))

		weights = torch.rand(10, 10)
		weights = torch.nn.Parameter(weights, requires_grad=True)
		reference_weights = torch.rand(10, 10)
		dale_law = DaleLawL2([weights], alpha=1, reference_weights=reference_weights)
		loss = dale_law()
		self.assertEqual(torch.round(loss.detach()), torch.round(torch.norm(weights.detach(), p="fro") ** 2))

		weights = torch.randn(10, 10)
		weights = torch.nn.Parameter(weights, requires_grad=True)
		dale_law = DaleLawL2([weights], alpha=1, reference_weights=None)
		loss = dale_law()
		self.assertEqual(torch.round(loss.detach()), torch.round(torch.norm(weights.detach(), p="fro") ** 2))

	def test_value_loss_t_random(self):
		"""
		Assert that the loss is computed correctly when t is random. The following scenario is:
		- Weights and reference weights are generated randomly.
		"""
		weights = torch.randn(10, 10)
		weights = torch.nn.Parameter(weights, requires_grad=True)
		reference_weights = torch.randn(10, 10)
		alpha = random.random()
		dale_law = DaleLawL2([weights], alpha, reference_weights)
		loss = dale_law()
		self.assertEqual(
			torch.round(loss.detach()),
			torch.round(
				torch.trace(weights.detach().T @ (alpha * weights.detach() - (1 - alpha) * torch.sign(reference_weights))))
		)


if __name__ == '__main__':
	unittest.main()
