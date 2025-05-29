import unittest
import warnings

from neurotorch.modules.layers import BaseNeuronsLayer
import torch


class TestBaseNeuronLayer(unittest.TestCase):

    def test_initalize_parameter(self):
        """
        Test if the parameters are initialized correctly.
        """
        layer = BaseNeuronsLayer()
        self.assertEqual(layer.input_size, None)
        self.assertEqual(layer.output_size, None)
        self.assertEqual(layer.name, "BaseNeuronsLayer")
        self.assertEqual(layer.use_recurrent_connection, True)
        self.assertEqual(layer.use_rec_eye_mask, False)
        self.assertEqual(layer.dt, 1e-3)

        layer = BaseNeuronsLayer(
            input_size=10,
            output_size=20,
            name="test",
            use_recurrent_connection=False,
            use_rec_eye_mask=False,
            freeze_weights=True,
            dt=1,
            device="cpu",
        )
        self.assertEqual(layer.input_size, 10)
        self.assertEqual(layer.output_size, 20)
        self.assertEqual(layer.name, "test")
        self.assertEqual(layer.use_recurrent_connection, False)
        self.assertEqual(layer.use_rec_eye_mask, False)
        self.assertEqual(layer.freeze_weights, True)
        self.assertEqual(layer.dt, 1)
        self.assertEqual(layer.device, "cpu")

    def test_not_implemented(self):
        """
        Test if the forward and create_empty_state are not implemented
        """
        layer = BaseNeuronsLayer()
        with self.assertRaises(NotImplementedError):
            layer.forward(None)

    def test_build(self):
        """
        Test if the build method works correctly. We always test the following matrix :
        1. forward_weights
        2. recurrent_weights
        3. recurrent_eye_mask
        """
        layer = BaseNeuronsLayer(
            100, 200, device="cpu", use_recurrent_connection=True, use_rec_eye_mask=True
        )
        self.assertEqual(layer.forward_weights, None)
        layer.build()
        self.assertIsInstance(layer.forward_weights, torch.Tensor)
        self.assertEqual(layer.forward_weights.shape, (100, 200))
        self.assertEqual(torch.round(layer.forward_weights.detach().mean()), 0)
        self.assertEqual(layer.forward_weights.requires_grad, True)
        self.assertEqual(layer.forward_weights.device.type, "cpu")
        self.assertIsInstance(layer.rec_mask, torch.Tensor)
        self.assertEqual(layer.rec_mask.device.type, "cpu")
        self.assertTrue(
            torch.isclose(torch.diag(layer.rec_mask, 0).sum(), torch.tensor(0.0))
        )
        self.assertIsInstance(layer.recurrent_weights, torch.Tensor)
        self.assertEqual(layer.recurrent_weights.device.type, "cpu")
        self.assertEqual(layer.recurrent_weights.requires_grad, True)

        layer = BaseNeuronsLayer(
            100,
            200,
            use_rec_eye_mask=False,
            use_recurrent_connection=False,
            device="cpu",
        )
        layer.build()
        self.assertIsInstance(layer.forward_weights, torch.Tensor)
        self.assertEqual(layer.forward_weights.shape, (100, 200))
        self.assertEqual(torch.round(layer.forward_weights.detach().mean()), 0)
        self.assertEqual(layer.forward_weights.requires_grad, True)
        self.assertEqual(layer.forward_weights.device.type, "cpu")
        self.assertEqual(layer.rec_mask, None)
        self.assertEqual(layer.recurrent_weights, None)

        if torch.cuda.is_available():
            layer = BaseNeuronsLayer(
                100,
                200,
                use_rec_eye_mask=False,
                use_recurrent_connection=True,
                device="cuda",
            )
            layer.build()
            self.assertIsInstance(layer.forward_weights, torch.Tensor)
            self.assertEqual(layer.forward_weights.shape, (100, 200))
            self.assertEqual(torch.round(layer.forward_weights.detach().mean()), 0)
            self.assertEqual(layer.forward_weights.requires_grad, True)
            self.assertEqual(layer.forward_weights.device.type, "cuda")
            self.assertIsInstance(layer.rec_mask, torch.Tensor)
            self.assertEqual(layer.rec_mask.device.type, "cuda")
            self.assertNotEqual(torch.diag(layer.rec_mask, 0).sum(), 0.0)
            self.assertIsInstance(layer.recurrent_weights, torch.Tensor)
            self.assertEqual(layer.recurrent_weights.device.type, "cuda")
        else:
            warnings.warn(
                "No CUDA available. Skipping test_build."
                "Please consider running the tests on a machine with CUDA.",
                UserWarning,
            )


if __name__ == "__main__":
    unittest.main()
