import random
import unittest

import torch

from neurotorch.regularisation.connectome import DaleLaw


class TestDaleLaw(unittest.TestCase):
	"""
	Test if the Dale's law is well implemented
	"""

	def test_size_output(self):
		"""
		Assert that the output is a 1D tensor
		"""
		weights_reference = torch.rand(10, 10)
		weights = torch.rand(10, 10)

		t = 0.5
		dale_law = DaleLaw(t, weights_reference)
		loss = dale_law(weights)
		self.assertEqual(loss.numel(), 1)

		t = 0
		dale_law = DaleLaw(t, weights_reference)
		loss = dale_law(weights)
		self.assertEqual(loss.numel(), 1)

		t = 1
		dale_law = DaleLaw(t, weights_reference)
		loss = dale_law(weights)
		self.assertEqual(loss.numel(), 1)

	def test_raise_error(self):
		"""
		Assert that the error is raised when needed
		"""
		with self.assertRaises(ValueError):
			weights_reference = torch.rand(10, 10)
			weights = torch.rand(10, 10)
			t = -1
			dale_law = DaleLaw(t, weights_reference)

		with self.assertRaises(ValueError):
			weights_reference = torch.rand(10, 10)
			weights = torch.rand(10, 10)
			t = 2
			dale_law = DaleLaw(t, weights_reference)

		with self.assertRaises(ValueError):
			weights = torch.rand(10, 10)
			t = 0
			dale_law = DaleLaw(t, reference_weights=None)

		try:
			dale_law = DaleLaw(t=1, reference_weights=None)
		except ValueError:
			self.fail("DaleLaw should not raise an error when weights_reference is None")

	def test_gradient_is_compute(self):
		"""
		Assert that the gradient is computed
		"""
		weights = torch.rand(10, 10)
		reference_weights = torch.rand(10, 10)
		weights = torch.nn.Parameter(weights, requires_grad=True)
		t = 0.5
		dale_law = DaleLaw(t, reference_weights)
		loss = dale_law(weights)
		loss.backward()
		self.assertTrue(weights.grad is not None)

		weights = torch.rand(10, 10)
		reference_weights = torch.rand(10, 10)
		weights = torch.nn.Parameter(weights, requires_grad=True)
		t = 0
		dale_law = DaleLaw(t, reference_weights)
		loss = dale_law(weights)
		loss.backward()
		self.assertTrue(weights.grad is not None)

		weights = torch.rand(10, 10)
		weights = torch.nn.Parameter(weights, requires_grad=True)
		t = 1
		dale_law = DaleLaw(t)
		loss = dale_law(weights)
		loss.backward()
		self.assertTrue(weights.grad is not None)

		weights = torch.rand(10, 10)
		reference_weights = torch.rand(10, 10)
		weights = torch.nn.Parameter(weights, requires_grad=True)
		t = 1
		dale_law = DaleLaw(t, reference_weights)
		loss = dale_law(weights)
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
		dale_law = DaleLaw(t=0, reference_weights=reference_weights)
		loss = dale_law(weights)
		self.assertEqual(loss, -torch.trace(weights.detach().T @ torch.sign(reference_weights)))

		weights = torch.rand(10, 10)
		weights = torch.nn.Parameter(weights, requires_grad=True)
		reference_weights = torch.rand(10, 10) * -1
		dale_law = DaleLaw(t=0, reference_weights=reference_weights)
		loss = dale_law(weights)
		self.assertEqual(loss, -torch.trace(weights.detach().T @ torch.sign(reference_weights)))
		loss_if_good = dale_law(weights * -1)
		self.assertTrue(loss_if_good < loss)

		weights = torch.randn(10, 10)
		weights = torch.nn.Parameter(weights, requires_grad=True)
		reference_weights = torch.randn(10, 10)
		dale_law = DaleLaw(t=0, reference_weights=reference_weights)
		loss = dale_law(weights)
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
		dale_law = DaleLaw(t=1, reference_weights=None)
		loss = dale_law(weights)
		self.assertEqual(torch.round(loss.detach()), torch.round(torch.norm(weights.detach(), p="fro") ** 2))

		weights = torch.rand(10, 10)
		weights = torch.nn.Parameter(weights, requires_grad=True)
		reference_weights = torch.rand(10, 10)
		dale_law = DaleLaw(t=1, reference_weights=reference_weights)
		loss = dale_law(weights)
		self.assertEqual(torch.round(loss.detach()), torch.round(torch.norm(weights.detach(), p="fro") ** 2))

		weights = torch.randn(10, 10)
		weights = torch.nn.Parameter(weights, requires_grad=True)
		dale_law = DaleLaw(t=1, reference_weights=None)
		loss = dale_law(weights)
		self.assertEqual(torch.round(loss.detach()), torch.round(torch.norm(weights.detach(), p="fro") ** 2))

	def test_value_loss_t_random(self):
		"""
		Assert that the loss is computed correctly when t is random. The following scenario is:
		- Weights and reference weights are generated randomly.
		"""
		weights = torch.randn(10, 10)
		weights = torch.nn.Parameter(weights, requires_grad=True)
		reference_weights = torch.randn(10, 10)
		t = random.random()
		dale_law = DaleLaw(t, reference_weights)
		loss = dale_law(weights)
		self.assertEqual(
			torch.round(loss.detach()),
			torch.round(
				torch.trace(weights.detach().T @ (t * weights.detach() - (1 - t) * torch.sign(reference_weights))))
		)


if __name__ == '__main__':
	unittest.main()
