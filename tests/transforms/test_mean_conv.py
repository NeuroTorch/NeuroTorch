import unittest
import torch
import numpy as np
from neurotorch.transforms.spikes_decoders import MeanConv


class TestMeanConv(unittest.TestCase):
	def test_output_shape(self):
		x = torch.rand(10, 8 * 16, 5)
		mean_conv = MeanConv(8)
		y = mean_conv(x)
		self.assertEqual(y.shape, (10, 16, 5))
		
	def test_output_shape_with_pad(self):
		x = torch.rand(10, 8 * 16 - 1, 5)
		mean_conv = MeanConv(8, pad_value=0.0, pad_mode="both")
		y = mean_conv(x)
		self.assertEqual(y.shape, (10, 16, 5))
		
		x = torch.rand(10, 8 * 16 + 1, 5)
		y = mean_conv(x)
		self.assertEqual(y.shape, (10, 17, 5))
		
		x = torch.rand(10, 8 * 16 - 2, 5)
		y = mean_conv(x)
		self.assertEqual(y.shape, (10, 16, 5))
		



