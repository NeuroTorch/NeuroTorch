import unittest

import torch

import neurotorch as nt

from ..mocks import MockTrainer


# class MockTrainer:
# 	def __init__(self, model):
# 		self.model = model
# 		self.current_training_state = CurrentTrainingState.get_null_state()
#
# 	def update_state_(self, **kwargs):
# 		self.current_training_state = self.current_training_state.update(**kwargs)
#
# 	def train(self, x_shape, model, callback):
# 		callback.start(self)
# 		x_rn = torch.rand(x_shape, device=model.device)
# 		pred, hh = model(x_rn)
# 		pred_tensor = nt.utils.maybe_unpack_singleton_dict(pred)
# 		with torch.no_grad():
# 			y = pred_tensor + 1
# 		batch_loss = torch.mean(torch.abs(pred_tensor - y))
# 		self.current_training_state = self.current_training_state.update(
# 			batch_loss=batch_loss, pred_batch=pred, y_batch=y
# 		)
# 		callback.on_optimization_begin(self)
# 		callback.on_optimization_end(self)


class TestBPTT(unittest.TestCase):
	def setUp(self) -> None:
		self.x_shape = (10, 10)
		self.network = nt.SequentialRNN(
			layers=[
				nt.LIFLayer(*self.x_shape),
			],
			device=torch.device("cpu"),
		).build()
		self.bptt = nt.BPTT()
		# self.trainer = MockTrainer(self.network)
		self.trainer = MockTrainer(callbacks=self.bptt)

	def test_start(self):
		self.bptt.start(self.trainer)
		for b_param, t_param in zip(self.bptt.params, self.trainer.model.parameters()):
			self.assertIs(b_param, t_param)
			
	def test_on_optimization_begin(self):
		initial_weights = [p.detach().cpu().clone() for p in self.trainer.model.parameters() if p.requires_grad]
		bptt = nt.BPTT()
		self.trainer.train(self.x_shape, self.network, bptt)
		final_weights = [p.detach().cpu().clone() for p in self.trainer.model.parameters() if p.requires_grad]
		for i, (i_w, f_w) in enumerate(zip(initial_weights, final_weights)):
			self.assertFalse(torch.allclose(i_w, f_w), f"Parameter {i} was not updated.")
			
	def test_on_optimization_end(self):
		bptt = nt.BPTT()
		self.trainer.train(self.x_shape, self.network, bptt)
		# check if the grad of the bptt's params is zero
		for p in bptt.params:
			self.assertTrue(torch.allclose(p.grad, torch.zeros_like(p.grad)))
		
		

