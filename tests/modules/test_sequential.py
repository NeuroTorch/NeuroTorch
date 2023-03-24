import unittest
import warnings
from typing import Iterable
from functools import partial

import numpy as np
import torch
from torchvision.transforms import Compose
from neurotorch import Sequential, Linear


class TestSequential(unittest.TestCase):
	def test_to(self):
		"""
		Test that the to method works as expected.
		:return: None
		"""
		model = Sequential(layers=[Linear(10, 10), Linear(10, 10)]).build()
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
		model = Sequential(layers=[Linear(10, 10), Linear(10, 10)]).build()
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












