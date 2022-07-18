import unittest

import torch

from neurotorch.transforms import LinearRateToSpikes, to_tensor


class TestLinearRateToSpikes(unittest.TestCase):
	def test_single_rate(self):
		n_steps = 10
		data_min = 0.0
		data_max = 1.0
		epsilon = 1e-6
		linear_to_spikes_rate = LinearRateToSpikes(
			n_steps=n_steps,
			data_min=data_min,
			data_max=data_max,
			epsilon=epsilon,
		)
		x = to_tensor(0.5)
		y_true = torch.zeros(n_steps, dtype=torch.float32)
		y_true[torch.floor((1-x) * n_steps).type(torch.int64)-1] = 1.0
		y_true[n_steps-1] = 1.0
		y_pred = linear_to_spikes_rate(x)
		self.assertTrue(torch.allclose(y_true, y_pred), msg=f"{x = }, {y_true} != {y_pred}")

		x = to_tensor(0.9)
		y_true = torch.ones(n_steps, dtype=torch.float32)
		y_true[0] = 0.0
		y_pred = linear_to_spikes_rate(x)
		self.assertTrue(torch.allclose(y_true, y_pred), msg=f"{x = }, {y_true} != {y_pred}")

		x = to_tensor(0.1)
		y_true = torch.zeros(n_steps, dtype=torch.float32)
		y_true[torch.floor((1 - x) * n_steps).type(torch.int64)-1] = 1.0
		y_pred = linear_to_spikes_rate(x)
		self.assertTrue(torch.allclose(y_true, y_pred), msg=f"{x = }, {y_true} != {y_pred}")

		x = to_tensor(data_min)
		y_true = torch.zeros(n_steps, dtype=torch.float32)
		y_true[n_steps-1] = 1.0
		y_pred = linear_to_spikes_rate(x)
		self.assertTrue(torch.allclose(y_true, y_pred), msg=f"{x = }, {y_true} != {y_pred}")

		x = to_tensor(data_max)
		y_true = torch.ones(n_steps, dtype=torch.float32)
		y_pred = linear_to_spikes_rate(x)
		self.assertTrue(torch.allclose(y_true, y_pred), msg=f"{x = }, {y_true} != {y_pred}")

	def test_batch(self):
		n_steps = 10
		data_min = 0.0
		data_max = 1.0
		epsilon = 1e-6
		linear_to_spikes_rate = LinearRateToSpikes(
			n_steps=n_steps,
			data_min=data_min,
			data_max=data_max,
			epsilon=epsilon,
		)
		x = to_tensor(torch.ones(8) * 0.5)
		y_true = torch.zeros(n_steps, 8, dtype=torch.float32)
		y_true[torch.floor((1-x) * n_steps).type(torch.int64)-1] = 1.0
		y_true[n_steps - 1] = 1.0
		y_pred = linear_to_spikes_rate(x)
		self.assertTrue(torch.allclose(y_true, y_pred), msg=f"{x = }, {y_true} != {y_pred}")

	def test_batch_2_dim(self):
		n_steps = 10
		data_min = 0.0
		data_max = 1.0
		epsilon = 1e-6
		linear_to_spikes_rate = LinearRateToSpikes(
			n_steps=n_steps,
			data_min=data_min,
			data_max=data_max,
			epsilon=epsilon,
		)
		x = to_tensor(torch.ones(8, 6) * 0.5)
		y_true = torch.zeros(n_steps, 8, 6, dtype=torch.float32)
		y_true[torch.floor((1-x) * n_steps).type(torch.int64)-1] = 1.0
		y_true[n_steps - 1] = 1.0
		y_pred = linear_to_spikes_rate(x)
		self.assertTrue(torch.allclose(y_true, y_pred), msg=f"{x = }, {y_true} != {y_pred}")


