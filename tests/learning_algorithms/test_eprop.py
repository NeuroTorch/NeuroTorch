import unittest

import torch

import neurotorch as nt

from ..mocks import MockTrainer


class TestEprop(unittest.TestCase):
    def setUp(self) -> None:
        self.eprop = nt.Eprop()
        self.in_layer = nt.Linear(10, 10).build()
        self.out_layer = nt.Linear(10, 5).build()
        self.model = nt.SequentialRNN(layers=[self.in_layer, self.out_layer]).build()
        self.trainer = MockTrainer(callbacks=self.eprop, model=self.model)

    def test_start(self):
        self.eprop.start(self.trainer)
        for b_param, t_param in zip(self.eprop.params, self.in_layer.parameters()):
            self.assertIs(b_param, t_param)
        for b_param, t_param in zip(self.eprop.output_params, self.out_layer.parameters()):
            self.assertIs(b_param, t_param)

    def test_on_optimization_begin(self):
        initial_weights = [p.detach().cpu().clone() for p in self.trainer.model.parameters() if p.requires_grad]
        self.trainer.train()
        final_weights = [p.detach().cpu().clone() for p in self.trainer.model.parameters() if p.requires_grad]
        for i, (i_w, f_w) in enumerate(zip(initial_weights, final_weights)):
            self.assertFalse(torch.allclose(i_w, f_w), f"Parameter {i} was not updated.")

    def test_on_optimization_end(self):
        self.trainer.train()
        # check if the grad of the eprop's params is zero
        for p in self.eprop.params:
            self.assertTrue(p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad)))
        for p in self.eprop.output_params:
            self.assertTrue(p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad)))
		
		

