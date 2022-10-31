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
	layer = model.get_layer()
	data = nt.to_tensor(data).T
	n_time_steps, n_units = data.shape
	# set up the training
	loss_function = nt.losses.PVarianceLoss()
	
	J = nt.to_tensor(1.5 * np.random.randn(n_units, n_units) / np.sqrt(n_units))
	layer.forward_weights = J
	
	ampInWN = kwargs.get("ampInWN", 0.01)
	dtRNN, tauWN = layer.dt, layer.tau
	ampWN = torch.sqrt(tauWN / dtRNN)
	iWN = ampWN * np.random.randn(n_time_steps, n_units)
	inputWN = torch.ones((n_time_steps, n_units))
	for tt in range(1, n_time_steps):
		inputWN[tt] = iWN[tt] + (inputWN[tt - 1] - iWN[tt]) * torch.exp(-(dtRNN / tauWN))
	inputWN = ampInWN * inputWN
	
	P = torch.eye(n_units)
	P_curbd = torch.eye(n_units)
	
	p_bar = tqdm.tqdm(range(kwargs.get("n_iterations", 10)))
	for iteration in p_bar:
		# curbd setup
		H = data[0, np.newaxis]
		y_pred_curbd = torch.zeros_like(data)
		y_pred_curbd[0] = np.tanh(H)
		
		# neurotorch setup
		y_pred = torch.zeros_like(data)
		y_pred[0] = data[0]
		hh = None
		for t in range(1, n_time_steps):
			# curbd step
			y_pred_curbd[t] = np.tanh(H)
			JR = torch.matmul(J, y_pred_curbd[t]) + inputWN[t]
			H = H + dtRNN * (JR - H) / tauWN
			J += curbd_step(y_pred_curbd[t], data[t], P_curbd)
			
			# neurotorch step
			y_pred[t], hh = layer(inputWN[t][np.newaxis, :], hh)
			layer.forward_weights.data += curbd_step(y_pred[t], data[t], P)
			
		# compute and print loss
		loss = loss_function(y_pred, data)
		loss_curbd = loss_function(y_pred_curbd, data)
		p_bar.set_description(f"Loss: {loss.item():.4f}, Loss curbd: {loss_curbd.item():.4f}")


if __name__ == '__main__':
	curbd_data = np.load("data/ts/curbd_Adata.npy")
	network = nt.SequentialModel(
		layers=[nt.WilsonCowanLayer(curbd_data.shape[0], curbd_data.shape[0], activation="tanh", tau=0.1)],
		foresight_time_steps=1,
		out_memory_size=1,
		hh_memory_size=1,
		device=torch.device("cpu"),
	).build()
	curbd_train(curbd_data, network)



