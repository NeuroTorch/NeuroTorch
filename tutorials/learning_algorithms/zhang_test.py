from collections import defaultdict
from copy import deepcopy
from typing import Optional, Tuple, NamedTuple

import torch
import neurotorch as nt
import numpy as np
import tqdm
from torch.nn import functional as F

from neurotorch.modules.layers import WilsonCowanCURBDLayer


def zhang_step(inputs, output, target, P, optimizer, **kwargs):
	labda = kwargs.get("labda", 0.9)
	kappa = kwargs.get("kappa", 0.1)
	eta = kwargs.get("eta", 1.0)
	optimizer.zero_grad()
	mse_loss = F.mse_loss(output.view(target.shape), target)
	mse_loss.backward()
	x = inputs.detach().clone().view(-1, 1)  # [ell, 1]
	u = torch.matmul(P, x)  # [ell, ell] @ [ell, 1] -> [ell, 1]
	h = labda + kappa * torch.matmul(x.T, u)  # [1, ell] @ [ell, 1] -> [1, 1]
	lr = (eta / h) * P
	for p in optimizer.param_groups[0]['params']:
		p.grad = torch.matmul(lr, p.grad)
	optimizer.step()
	P = (1/labda)*P - (kappa/(labda*h))*torch.matmul(u, u.T)  # [ell, 1] @ [1, ell] -> [ell, ell]
	return P


def zhang_train(data, model, **kwargs):
	nt.set_seed(kwargs.get('seed', 0))
	layer = deepcopy(model.get_layer())
	
	data = nt.to_tensor(data).T
	n_time_steps, n_units = data.shape
	
	# set up the training
	loss_function = nt.losses.PVarianceLoss()
	# J = nt.to_tensor(1.5 * np.random.randn(n_units, n_units) / np.sqrt(n_units))
	# layer.forward_weights = J.clone()
	# model.get_layer().forward_weights = J.clone()
	
	optimizer = torch.optim.SGD([layer.forward_weights], lr=kwargs.get("lr", 1.0))
	
	# set up the learning algorithm
	# trainer = nt.Trainer(model)
	# learning_algorithm = WeakRLS(params=[model.get_layer().forward_weights])
	# learning_algorithm.start(trainer)
	
	P_nt = torch.eye(n_units)
	losses = defaultdict(list)
	
	p_bar = tqdm.tqdm(range(kwargs.get("n_iterations", 10)))
	for iteration in p_bar:
		# nt.layer setup
		y_pred_nt = torch.zeros_like(data)
		hh_nt = None
		y_pred_nt[0] = torch.tanh(data[0])
		
		# seq setup
		# y_pred_seq = torch.zeros_like(data)
		# y_pred_seq[0] = torch.tanh(data[0])
		# x_batch = y_pred_seq[0][np.newaxis, np.newaxis, :]
		# y_batch = data[np.newaxis, :]
		# trainer.update_state_(x_batch=x_batch, y_batch=y_batch)
		# learning_algorithm.on_batch_begin(trainer)
		# y_pred_seq[1:] = model.get_prediction_trace(x_batch)
		# learning_algorithm.on_batch_end(trainer)
		
		for t in range(1, n_time_steps):
			# nt.layer step
			inputs = y_pred_nt[t-1][np.newaxis, :].detach().clone()
			out, hh_nt = layer(inputs, hh_nt)
			# activation = layer.activation(torch.matmul(y_pred_nt[t-1][np.newaxis, :], layer.forward_weights) - layer.mu)
			# ratio_dt_tau = layer.dt / layer.tau
			# transition_rate = (1 - hh_nt * self.r)
			# out = hh_nt * (1 - ratio_dt_tau) + transition_rate * activation * ratio_dt_tau
			y_pred_nt[t] = out.detach().clone()
			P_nt = zhang_step(inputs, out, data[t], P_nt, optimizer)
		
		# compute and print loss
		loss_nt = loss_function(y_pred_nt, data)
		# loss_seq = loss_function(y_pred_seq, data)
		losses['nt.layer'].append(loss_nt.item())
		# losses['seq'].append(loss_seq.item())
		p_bar.set_description(
			f"Loss nt.layer: {loss_nt.item():.4f}, "
			# f"Loss seq: {loss_seq.item():.4f} "
		)


if __name__ == '__main__':
	curbd_data = np.load("data/ts/curbd_Adata.npy")
	network = nt.SequentialModel(
		layers=[nt.WilsonCowanLayer(
			curbd_data.shape[0], curbd_data.shape[0],
			activation="sigmoid",
			tau=0.1,
			dt=0.01,
		)],
		# layers=[WilsonCowanCURBDLayer(
		# 	curbd_data.shape[0], curbd_data.shape[0],
		# 	activation="tanh",
		# 	tau=0.1,
		# 	dt=0.01,
		# )],
		foresight_time_steps=curbd_data.shape[-1]-1,
		out_memory_size=curbd_data.shape[-1]-1,
		hh_memory_size=1,
		device=torch.device("cpu"),
	).build()
	zhang_train(curbd_data, network)



