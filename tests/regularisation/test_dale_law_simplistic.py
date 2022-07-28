import unittest
import torch
from neurotorch.regularisation.connectome import DaleLawSimplistic


class TestDaleLawSimplistic(unittest.TestCase):
	"""
	Test if the Dale's law is well implemented.
	"""
	def test_size_output(self):
		"""
		Assert that the output is a 1D tensor.
		"""
		weights = torch.rand(10, 10)
		dale_law = DaleLawSimplistic()
		loss = dale_law(weights)
		self.assertEqual(loss.numel(), 1)

	def test_gradient_is_compute(self):
		"""
		Assert that the gradient is computed.
		"""
		weights = torch.rand(10, 10)
		weights = torch.nn.Parameter(weights, requires_grad=True)
		dale_law = DaleLawSimplistic()
		loss = dale_law(weights)
		loss.backward()
		self.assertTrue(weights.grad is not None)

	def test_value_loss(self):
		"""
		Assert that the loss is computed correctly. The following scenario are:
		- Neurons are all excitatory. A loss of 0 is expected
		- Neurons are all inhibitory. A loss of 0 is expected
		- Neurons have hald excitatory and half inhibitory connexion. A loss of 1 is expected
		- Neurons have 3/4 excitatory and 1/4 inhibitory connexion. A loss of 0.5 is expected
		- Neurons have all excitatory connexion, some connexions are not present. A loss of 0 is expected
		- Neurons have no connexion. A loss of 0 is expected
		"""
		weights = torch.rand(10, 10)
		weights = torch.nn.Parameter(weights, requires_grad=True)
		dale_law = DaleLawSimplistic()
		loss = dale_law(weights)
		self.assertEqual(loss.item(), 0)

		weights = torch.rand(10, 10) * -1
		weights = torch.nn.Parameter(weights, requires_grad=True)
		dale_law = DaleLawSimplistic()
		loss = dale_law(weights)
		self.assertEqual(loss.item(), 0)

		weights = torch.rand(10, 10)
		weights[5:, :] = weights[5:, :] * -1
		weights = torch.nn.Parameter(weights, requires_grad=True)
		dale_law = DaleLawSimplistic()
		loss = dale_law(weights)
		self.assertEqual(loss.item(), 1)

		weights = torch.rand(100, 100)
		weights[::4, :] = weights[::4, :] * -1
		weights = torch.nn.Parameter(weights, requires_grad=True)
		dale_law = DaleLawSimplistic()
		loss = dale_law(weights)
		self.assertEqual(loss.item(), 0.5)


		weights = torch.rand(10, 10) * -1
		weights.fill_diagonal_(0)
		weights = torch.nn.Parameter(weights, requires_grad=True)
		dale_law = DaleLawSimplistic()
		loss = dale_law(weights)
		self.assertEqual(loss.item(), 0)

		weights = torch.zeros(10, 10)
		weights = torch.nn.Parameter(weights, requires_grad=True)
		dale_law = DaleLawSimplistic()
		loss = dale_law(weights)
		self.assertEqual(loss.item(), 0)


if __name__ == '__main__':
	unittest.main()
