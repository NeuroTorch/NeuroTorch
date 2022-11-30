import unittest
import torch

from neurotorch.utils import batchwise_temporal_filter, batchwise_temporal_recursive_filter


class TestUtils(unittest.TestCase):
	def test_batchwise_temporal_filter(self):
		def conv_filter(x, decay):
			batch_size, time_steps, *_ = x.shape
			assert time_steps >= 1
			weighs = torch.tensor([decay ** t for t in range(time_steps)], dtype=torch.float32)
			y = torch.nn.functional.conv1d(x, weighs.unsqueeze(0).unsqueeze(0))
			return y
		
		x = torch.arange(3).unsqueeze(0).unsqueeze(-1).float()
		alpha = 0.1
		target = batchwise_temporal_recursive_filter(x, alpha)
		# conv_ = conv_filter(x, alpha)
		y = batchwise_temporal_filter(x, alpha)
		self.assertTrue(torch.allclose(y, target), f"{torch.cat([x, y, target, ], dim=-1)}")
	