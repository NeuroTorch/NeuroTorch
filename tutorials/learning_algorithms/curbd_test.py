from collections import defaultdict
from copy import deepcopy

import torch
import neurotorch as nt
import numpy as np
import tqdm


def curbd_step(output, target, P):
	out_view = output.view(-1, 1)
	error = output - target
	k = torch.matmul(P, out_view)
	outPout = torch.matmul(out_view.T, k).item()
	c = 1 / (1 + outPout)
	P -= c * torch.matmul(k, k.T)
	return -c*torch.outer(error.view(-1), k.view(-1))


def curbd_train(data, model, **kwargs):
	nt.set_seed(kwargs.get('seed', 0))
	layer = model.get_layer()
	layer_mod = deepcopy(layer)
	
	data = nt.to_tensor(data).T
	n_time_steps, n_units = data.shape
	
	# set up the training
	loss_function = nt.losses.PVarianceLoss()
	J = nt.to_tensor(1.5 * np.random.randn(n_units, n_units) / np.sqrt(n_units))
	layer.forward_weights = J.detach().clone()
	layer_mod.forward_weights = J.detach().clone()
	
	# set up the curbd input
	ampInWN = kwargs.get("ampInWN", 0.01)
	dtRNN, tauWN = layer.dt, layer.tau
	ampWN = torch.sqrt(tauWN / dtRNN)
	iWN = ampWN * np.random.randn(n_time_steps, n_units)
	inputWN = torch.ones((n_time_steps, n_units))
	for tt in range(1, n_time_steps):
		inputWN[tt] = iWN[tt] + (inputWN[tt - 1] - iWN[tt]) * torch.exp(-(dtRNN / tauWN))
	inputWN = ampInWN * inputWN
	
	P_nt = torch.eye(n_units)
	P_nt_mod = torch.eye(n_units)
	P_curbd = torch.eye(n_units)
	losses = defaultdict(list)
	
	p_bar = tqdm.tqdm(range(kwargs.get("n_iterations", 10)))
	for iteration in p_bar:
		# curbd setup
		y_pred_curbd = torch.zeros_like(data)
		hh_curbd = data[0, np.newaxis]
		y_pred_curbd[0] = torch.tanh(hh_curbd)
		
		# neurotorch mod setup
		y_pred_nt_mod = torch.zeros_like(data)
		hh_nt_mod = data[0]
		y_pred_nt_mod[0] = torch.tanh(hh_nt_mod)
		
		# neurotorch setup
		y_pred_nt = torch.zeros_like(data)
		hh_nt = None
		y_pred_nt[0] = data[0]
		
		for t in range(1, n_time_steps):
			# curbd step
			y_pred_curbd[t] = torch.tanh(hh_curbd)
			JR_curbd = torch.matmul(J, y_pred_curbd[t]) + inputWN[t]
			hh_curbd = hh_curbd + dtRNN * (JR_curbd - hh_curbd) / tauWN
			J += curbd_step(y_pred_curbd[t], data[t], P_curbd)
			
			# neurotorch mod step
			y_pred_nt_mod[t] = torch.tanh(hh_nt_mod)
			JR_nt_mod = torch.matmul(layer_mod.forward_weights, y_pred_nt_mod[t]) + inputWN[t]
			hh_nt_mod = hh_nt_mod + layer_mod.dt * (JR_nt_mod - hh_nt_mod) / layer_mod.tau
			layer_mod.forward_weights.data += curbd_step(y_pred_nt_mod[t], data[t], P_nt_mod)
			
			# neurotorch step
			y_pred_nt[t], hh_nt = layer(inputWN[t][np.newaxis, :], hh_nt)
			layer.forward_weights.data += curbd_step(y_pred_nt[t], data[t], P_nt)
		
		# compute and print loss
		loss_nt = loss_function(y_pred_nt, data)
		loss_nt_mod = loss_function(y_pred_nt_mod, data)
		loss_curbd = loss_function(y_pred_curbd, data)
		losses['nt'].append(loss_nt.item())
		losses['nt_mod'].append(loss_nt_mod.item())
		losses['curbd'].append(loss_curbd.item())
		p_bar.set_description(
			f"Loss nt: {loss_nt.item():.4f}, "
			f"Loss curbd: {loss_curbd.item():.4f}, "
			f"Loss nt mod: {loss_nt_mod.item():.4f}"
		)


if __name__ == '__main__':
	curbd_data = np.load("data/ts/curbd_Adata.npy")
	network = nt.SequentialModel(
		layers=[nt.WilsonCowanLayer(
			curbd_data.shape[0], curbd_data.shape[0],
			activation="tanh",
			tau=0.1,
			dt=0.01,
		)],
		foresight_time_steps=1,
		out_memory_size=1,
		hh_memory_size=1,
		device=torch.device("cpu"),
	).build()
	curbd_train(curbd_data, network)



