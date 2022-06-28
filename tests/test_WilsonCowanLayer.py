import unittest

import torch
import numpy as np

from src.neurotorch.modules.layers import WilsonCowanLayer


class WilsonCowanLayerTest(unittest.TestCase):
    # TODO: test_backward, test_mu (if array, tensor or scalar), test input on wrong device
    def test_kwargs_default_parameter(self):
        """
        Test if the parameter are well initialize in the kwargs
        """
        # If unspecified
        layer = WilsonCowanLayer()
        self.assertEqual(layer.std_weight, 1.0)
        self.assertEqual(layer.mu, 0.0)
        self.assertEqual(layer.tau, 1.0)
        self.assertEqual(layer.kwargs["learn_mu"], False)
        self.assertEqual(layer.mean_mu, 2.0)
        self.assertEqual(layer.std_mu, 0.0)

        # If specified
        layer = WilsonCowanLayer(std_weight=10.0, mu=-2.0, tau=3.0, learn_mu=True, mean_mu=4.0, std_mu=5.0)
        self.assertEqual(layer.std_weight, 10.0)
        self.assertEqual(layer.mu, -2.0)
        self.assertEqual(layer.tau, 3.0)
        self.assertEqual(layer.mean_mu, 4.0)
        self.assertEqual(layer.std_mu, 5.0)
        self.assertEqual(layer.kwargs["learn_mu"], True)

    def test_instance(self):
        """
        Test if the variables are initialized with the correct instance
        """
        layer = WilsonCowanLayer()
        self.assertIsInstance(layer.std_weight, float)
        self.assertIsInstance(layer.mu, torch.Tensor)
        self.assertIsInstance(layer.mean_mu, float)
        self.assertIsInstance(layer.std_mu, float)
        self.assertIsInstance(layer.tau, float)
        self.assertIsInstance(layer.kwargs["learn_mu"], bool)
        self.assertIs(layer.input_size, None)
        self.assertIs(layer.output_size, None)
        self.assertIsInstance(layer.dt, float)

    def test_if_grad(self):  # TODO: add training to r
        """
        Test if the gradient is computed correctly for the parameters
        """
        layer = WilsonCowanLayer(input_size=10, output_size=10, learn_mu=False)
        layer.build()
        self.assertIs(layer.mu.requires_grad, False)
        self.assertIs(layer.forward_weights.requires_grad, True)
        layer = WilsonCowanLayer(input_size=10, output_size=10, learn_mu=True)
        layer.build()
        self.assertIs(layer.mu.requires_grad, True)
        self.assertIs(layer.forward_weights.requires_grad, True)

    def test_mu_r_dtype(self):
        """
        Test if the mu and r are of the correct dtype even if the user gives a wrong dtype
        """
        #TODO: add test for r
        layer = WilsonCowanLayer(input_size=10, output_size=10)
        layer.build()
        self.assertEqual(layer.mu.dtype, torch.float32)

        mu = np.random.rand(10).astype(np.float64)
        layer = WilsonCowanLayer(input_size=10, output_size=10, mu=mu)
        self.assertEqual(layer.mu.dtype, torch.float32)

        mu = torch.tensor(2.0, dtype=torch.float64)
        layer = WilsonCowanLayer(input_size=10, output_size=10, mu=torch.tensor(2.0, dtype=torch.float64))
        self.assertEqual(layer.mu.dtype, torch.float32)


    def test_device(self):  # TODO: test CPU + add device to input
        """
        Test if the layer is working on the correct device. The input is being place on the wrong device
        """
        layer = WilsonCowanLayer(input_size=3, output_size=3)
        input_ = torch.rand(3, 1, device="cpu")
        output = layer(input_)
        self.assertIsInstance(output[0], torch.Tensor)
        self.assertEqual(output[1], None)
        self.assertEqual(output[0].shape, (3, 1))
        self.assertEqual(output[0].device.type, layer.device.type)

        layer = WilsonCowanLayer(input_size=3, output_size=3, device="cuda:0")
        input_ = torch.rand(3, 1, device="cuda")
        output = layer(input_)
        self.assertIsInstance(output[0], torch.Tensor)
        self.assertEqual(output[1], None)
        self.assertEqual(output[0].shape, (3, 1))
        self.assertEqual(output[0].device, torch.device("cuda", index=0))

        layer = WilsonCowanLayer(input_size=3, output_size=3, device="cpu")
        input_ = torch.rand(3, 1, device="cpu")
        output = layer(input_)
        self.assertIsInstance(output[0], torch.Tensor)
        self.assertEqual(output[1], None)
        self.assertEqual(output[0].shape, (3, 1))
        self.assertEqual(output[0].device, torch.device("cpu"))

    def test_forward(self):
        """
        Test if the forward method works correctly
        """
        layer = WilsonCowanLayer(input_size=3, output_size=3)
        input_ = torch.rand(3, 1)
        output = layer(input_)
        self.assertIsInstance(output[0], torch.Tensor)
        self.assertEqual(output[1], None)
        self.assertEqual(output[0].shape, (3, 1))

    def test_intialize_weight(self):
        """
        Test if the random weight is initialize with the correct size, mean and STD
        """
        layer = WilsonCowanLayer(input_size=50, output_size=50, std_weight=6.0)
        layer.build()
        self.assertEqual(torch.round(layer.forward_weights.detach().mean()), 0.0)
        self.assertEqual(torch.round(layer.forward_weights.detach().std()), 6.0)

    def test_backward(self):   #TODO: torch.grad == None
        """
        Test if the backward method works correctly
        """
        #layer = WilsonCowanLayer(input_size=3, output_size=3, device="cpu")
        #input_ = torch.rand(3, 1)
        #output = layer(input_)[0]
        #output.mean().backward()
        #self.assertIsInstance(output.grad, torch.Tensor)
        #self.assertEqual(output.grad.shape, (3, 1))
        #self.assertEqual(output.grad.device, torch.device("cpu"))

        #layer = WilsonCowanLayer(input_size=3, output_size=3, device="cuda")
        #input_ = torch.rand(3, 1)
        #output = layer(input_)[0]
        #output.mean().backward()
        #self.assertIsInstance(output.grad, torch.Tensor)
        #self.assertEqual(output.grad.shape, (3, 1))
        #self.assertEqual(output.grad.device, torch.device("cuda", index=0))

    def test_output_result(self):
        """
        Test if the output result match the true data
        """
        layer = WilsonCowanLayer(input_size=60, output_size=60, device="cpu")
        input_ = torch.rand(60, 1, device=layer.device)
        output = layer(input_)[0]
        ratio_dt_tau = layer.dt / layer.tau
        sigmoid = (torch.sigmoid(torch.matmul(layer.forward_weights, input_) - layer.mu))
        true_output = input_ * (1 - ratio_dt_tau) + ratio_dt_tau * sigmoid
        self.assertEqual(output.all(), true_output.all())

        mu = torch.rand(60, 1, device="cpu")
        layer = WilsonCowanLayer(input_size=60, output_size=60, mu=mu, device="cpu")
        input_ = torch.rand(60, 1, device=layer.device)
        output = layer(input_)[0]
        ratio_dt_tau = layer.dt / layer.tau
        sigmoid = (torch.sigmoid(torch.matmul(layer.forward_weights, input_) - layer.mu))
        true_output = input_ * (1 - ratio_dt_tau) + ratio_dt_tau * sigmoid
        self.assertEqual(output.all(), true_output.all())


if __name__ == '__main__':
    unittest.main()