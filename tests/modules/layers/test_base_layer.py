import unittest
import warnings

import numpy as np
import torch
from neurotorch.dimension import Dimension

from neurotorch.modules.layers import BaseLayer


class TestBaseLayer(unittest.TestCase):

	def test_property(self):
		"""
		Test if the property of BaseLayer are initialized correctly.
		"""
		layer = BaseLayer(10, 10, name="test", device="cpu")
		layer.build()
		self.assertEqual(layer.input_size, 10)
		self.assertEqual(layer.output_size, 10)
		self.assertEqual(layer.requires_grad, True)
		self.assertEqual(layer.name, "test")
		self.assertEqual(layer._name_is_set, True)
		self.assertEqual(layer.is_ready_to_build, True)
		self.assertEqual(layer.is_built, True)
		self.assertEqual(layer.device, "cpu")

		if torch.cuda.is_available():
			layer = BaseLayer(device="cuda", learning_type=2)
			self.assertEqual(layer.device, "cuda")
			self.assertEqual(layer.requires_grad, False)
			self.assertEqual(layer.input_size, None)
			self.assertEqual(layer.output_size, None)
			self.assertEqual(layer.name_is_set, False)
			self.assertEqual(layer.name, "BaseLayer")
		else:
			warnings.warn(
				"No CUDA available. Skipping test_property. Please consider running the tests on a machine with CUDA.",
				UserWarning,
			)

	def test_isinstance(self):
		"""
		Test if the arguments of BaseLayer are initialized in the correct instance.
		"""
		layer = BaseLayer(10, 10, name="test", device="cpu")
		self.assertIsInstance(layer.input_size, Dimension)
		self.assertIsInstance(layer.output_size, Dimension)
		self.assertIsInstance(layer.name, str)
		self.assertIsInstance(layer.device, str)

	def test_not_implemented_error(self):
		"""
		Test if forward and create_empty_state are not implemented
		:return:
		"""
		with self.assertRaises(NotImplementedError):
			layer = BaseLayer()
			layer.forward(None)
		with self.assertRaises(NotImplementedError):
			layer = BaseLayer()
			layer.create_empty_state()

	def test_setter(self):
		"""
		Test if the setter of BaseLayer works correctly.
		"""
		layer = BaseLayer()
		self.assertEqual(layer._format_size(None), None)
		layer.input_size = 10
		layer.output_size = 15
		self.assertEqual(layer._format_size(1), 1)
		self.assertEqual(layer.input_size, 10)
		self.assertEqual(layer.output_size, 15)
		layer.name = "This is a test"
		self.assertEqual(layer.name, "This is a test")

	def test_build_sequence(self):
		"""
		Test if the sequence of build is correct.
		"""
		layer = BaseLayer(10, 10)
		self.assertEqual(layer._is_built, False)
		layer.build()
		self.assertEqual(layer._is_built, True)
		with self.assertRaises(ValueError) as context:
			layer.build()

	def test_update_regularization(self):
		"""
		Test if the update of regularization is correct.
		"""
		layer = BaseLayer(10, 10, device=torch.device("cpu"))
		layer.build()

		def _update(x):
			layer._regularization_loss = torch.tensor(x)
			return layer._regularization_loss

		layer.update_regularization_loss = _update
		layer.update_regularization_loss(0.1)
		self.assertTrue(
			torch.isclose(layer._regularization_loss, torch.tensor(0.1)),
			f"{layer._regularization_loss = } != {torch.tensor(0.1)}"
		)

	def test_reset_regularization(self):
		layer = BaseLayer(10, 10, device=torch.device("cpu"))
		layer.build()

		def _update(x):
			layer._regularization_loss = torch.tensor(x)
			return layer._regularization_loss

		layer.update_regularization_loss = _update
		layer.update_regularization_loss(0.1)
		self.assertTrue(torch.isclose(layer._regularization_loss, torch.tensor(0.1)))
		layer.reset_regularization_loss()
		self.assertTrue(torch.isclose(layer._regularization_loss, torch.tensor(0.0)))

	def test_get_regularization(self):
		layer = BaseLayer(10, 10, device=torch.device("cpu"))
		layer.build()

		def _update(x):
			layer._regularization_loss = torch.tensor(x)
			return layer._regularization_loss

		layer.update_regularization_loss = _update
		layer.update_regularization_loss(0.1)
		self.assertTrue(torch.isclose(layer._regularization_loss, layer.get_regularization_loss()))

	def test_get_and_reset_regularization(self):
		layer = BaseLayer(10, 10, device=torch.device("cpu"))
		layer.build()

		def _update(x):
			layer._regularization_loss = torch.tensor(x)
			return layer._regularization_loss

		layer.update_regularization_loss = _update
		layer.update_regularization_loss(0.1)
		self.assertTrue(torch.isclose(layer._regularization_loss, torch.tensor(0.1)))
		get_tensor = layer.get_regularization_loss()
		self.assertTrue(torch.isclose(layer._regularization_loss, get_tensor))
		self.assertTrue(torch.isclose(get_tensor, layer.get_and_reset_regularization_loss()))
		self.assertTrue(torch.isclose(layer._regularization_loss, torch.tensor(0.0)))


if __name__ == '__main__':
	unittest.main()
