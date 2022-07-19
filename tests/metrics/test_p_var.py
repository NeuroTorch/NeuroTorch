import unittest
import torch
from neurotorch.metrics.regression import RegressionMetrics


class TestpVar(unittest.TestCase):

	def test_return_scalar(self):
		y_pred = torch.rand(1, 2, 10)
		y_true = torch.rand(1, 2, 10)
		p_var = RegressionMetrics.compute_p_var(y_true, y_pred, torch.device('cpu'))
		self.assertEqual(p_var.numel(), 1)
		self.assertIsInstance(p_var, torch.Tensor)

	def test_result_one_batch_manual_MSE(self):
		y_true = torch.rand(1, 2, 10)
		y_pred = torch.rand(1, 2, 10)
		MSE = 1/20 * torch.sum((y_true - y_pred) ** 2)
		Var = torch.var(y_true)
		p_var = 1 - MSE / Var
		self.assertAlmostEqual(p_var.numpy(), RegressionMetrics.compute_p_var(y_true, y_pred, torch.device('cpu')).numpy())

	def test_result_one_batch_built_in_MSE(self):
		y_true = torch.rand(1, 2, 10)
		y_pred = torch.rand(1, 2, 10)
		MSE = torch.nn.MSELoss()
		MSE = MSE(y_true, y_pred)
		Var = torch.var(y_true)
		p_var = 1 - MSE / Var
		self.assertAlmostEqual(p_var.numpy(), RegressionMetrics.compute_p_var(y_true, y_pred, torch.device('cpu')).numpy())

	def test_result_multiple_batch_mean(self):
		y_true_1 = torch.rand(1, 2, 10)
		y_true_2 = torch.rand(1, 2, 10)
		y_pred_1 = torch.rand(1, 2, 10)
		y_pred_2 = torch.rand(1, 2, 10)

		p_var_1 = RegressionMetrics.compute_p_var(y_true_1, y_pred_1, torch.device('cpu'))
		p_var_2 = RegressionMetrics.compute_p_var(y_true_2, y_pred_2, torch.device('cpu'))

		p_var_mean = (p_var_1 + p_var_2) / 2
		self.assertAlmostEqual(
			p_var_mean.numpy(),
			RegressionMetrics.compute_p_var(torch.cat((y_true_1, y_true_2)), torch.cat((y_pred_1, y_pred_2)), torch.device('cpu')).numpy()
		)

	def test_result_multiple_batch_built_in_sum(self):
		y_true_1 = torch.rand(1, 2, 10)
		y_true_2 = torch.rand(1, 2, 10)
		y_pred_1 = torch.rand(1, 2, 10)
		y_pred_2 = torch.rand(1, 2, 10)

		p_var_1 = RegressionMetrics.compute_p_var(y_true_1, y_pred_1, torch.device('cpu'))
		p_var_2 = RegressionMetrics.compute_p_var(y_true_2, y_pred_2, torch.device('cpu'))
		p_var_sum = torch.sum(p_var_1 + p_var_2)
		self.assertAlmostEqual(
			p_var_sum.numpy(),
			RegressionMetrics.compute_p_var(
				torch.cat((y_true_1, y_true_2)), torch.cat((y_pred_1, y_pred_2)),
				torch.device('cpu'), reduction="sum"
			).numpy()
		)

	def test_result_multiple_batch_built_in_none(self):
		y_true_1 = torch.rand(1, 2, 10)
		y_true_2 = torch.rand(1, 2, 10)
		y_pred_1 = torch.rand(1, 2, 10)
		y_pred_2 = torch.rand(1, 2, 10)
		p_var_1 = RegressionMetrics.compute_p_var(y_true_1, y_pred_1, torch.device('cpu'))
		p_var_2 = RegressionMetrics.compute_p_var(y_true_2, y_pred_2, torch.device('cpu'))
		p_var = torch.Tensor([p_var_1, p_var_2])
		self.assertTrue(
			torch.allclose(
				p_var,
				RegressionMetrics.compute_p_var(
					torch.cat((y_true_1, y_true_2)), torch.cat((y_pred_1, y_pred_2)),
					torch.device('cpu'), reduction="none",
				)
			)
		)


if __name__ == '__main__':
	unittest.main()
