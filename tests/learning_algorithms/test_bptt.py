import unittest

import torch

import neurotorch as nt

from ..mocks import MockTrainer


class TestBPTT(unittest.TestCase):
	def setUp(self) -> None:
		self.bptt = nt.BPTT()
		self.trainer = MockTrainer(callbacks=self.bptt)

	def test_start(self):
		self.bptt.start(self.trainer)
		for b_param, t_param in zip(self.bptt.params, self.trainer.model.parameters()):
			self.assertIs(b_param, t_param)
			
	def test_on_optimization_begin(self):
		initial_weights = [p.detach().cpu().clone() for p in self.trainer.model.parameters() if p.requires_grad]
		self.trainer.train()
		final_weights = [p.detach().cpu().clone() for p in self.trainer.model.parameters() if p.requires_grad]
		for i, (i_w, f_w) in enumerate(zip(initial_weights, final_weights)):
			self.assertFalse(torch.allclose(i_w, f_w), f"Parameter {i} was not updated.")

	def test_on_optimization_end(self):
		self.trainer.train()
		# check if the grad of the bptt's params is zero
		for p in self.bptt.params:
			self.assertTrue(p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad)))
		
		

