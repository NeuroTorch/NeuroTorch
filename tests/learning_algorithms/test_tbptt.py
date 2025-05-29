import unittest

import torch

import neurotorch as nt

from ..mocks import MockTrainer


class TestTBPTT(unittest.TestCase):
    def setUp(self) -> None:
        self.tbptt = nt.TBPTT()
        self.trainer = MockTrainer(callbacks=self.tbptt)

    def test_start(self):
        self.tbptt.start(self.trainer)
        for b_param, t_param in zip(self.tbptt.params, self.trainer.model.parameters()):
            self.assertIs(b_param, t_param)

    def test_on_optimization_begin(self):
        initial_weights = [
            p.detach().cpu().clone()
            for p in self.trainer.model.parameters()
            if p.requires_grad
        ]
        self.trainer.train()
        final_weights = [
            p.detach().cpu().clone()
            for p in self.trainer.model.parameters()
            if p.requires_grad
        ]
        for i, (i_w, f_w) in enumerate(zip(initial_weights, final_weights)):
            self.assertFalse(
                torch.allclose(i_w, f_w), f"Parameter {i} was not updated."
            )

    def test_on_optimization_end(self):
        self.trainer.train()
        # check if the grad of the bptt's params is zero
        for p in self.tbptt.params:
            self.assertTrue(
                p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad))
            )

    def test_forward_2_linear_layers(self):
        self.trainer.model = nt.SequentialRNN(
            layers=[
                nt.Linear(self.trainer.x_shape[-1], self.trainer.x_shape[-1]),
                nt.Linear(self.trainer.x_shape[-1], self.trainer.x_shape[-1]),
            ]
        )
        self.trainer.y_shape = self.trainer.x_shape
        self.trainer.train()
        # check if the grad of the bptt's params is zero
        for p in self.tbptt.params:
            self.assertTrue(
                p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad))
            )

    def test_forward_2_wc_layers(self):
        self.trainer.x_shape = (1, 10, 5)
        self.trainer.model = nt.SequentialRNN(
            layers=[
                nt.WilsonCowanLayer(self.trainer.x_shape[-1], self.trainer.x_shape[-1]),
                nt.WilsonCowanLayer(self.trainer.x_shape[-1], self.trainer.x_shape[-1]),
            ]
        )
        self.trainer.y_shape = self.trainer.x_shape
        self.trainer.train()
        # check if the grad of the bptt's params is zero
        for p in self.tbptt.params:
            self.assertTrue(
                p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad))
            )
