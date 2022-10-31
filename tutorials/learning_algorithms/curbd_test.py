import torch
import neurotorch as nt
import numpy as np
import tqdm


def curbd_train(data, model, **kwargs):
	data = nt.to_tensor(data).T
	n_time_steps, n_units = data.shape
	# set up the training
	loss_function = nt.losses.PVarianceLoss()
	
	# ampWN = np.sqrt(tauWN / dtRNN)
	# iWN = ampWN * npr.randn(n_units, n_time_steps)
	# inputWN = np.ones((n_units, n_time_steps))
	# for tt in range(1, n_time_steps):
	# 	inputWN[:, tt] = iWN[:, tt] + (inputWN[:, tt - 1] - iWN[:, tt]) * np.exp(- (dtRNN / tauWN))
	# inputWN = ampInWN * inputWN
	
	P = torch.eye(n_units)
	p_bar = tqdm.tqdm(range(kwargs.get("n_iterations", 100)))
	for iteration in p_bar:
		y_pred = torch.zeros_like(data)
		y_pred[0] = data[0]
		for t in range(1, n_time_steps):
			y_pred[t] = model.get_prediction_trace(y_pred[t-1][np.newaxis, np.newaxis, :]).squeeze(0)
			
			out_view = y_pred[t].view(-1, 1)
			error = y_pred[t] - data[t]
			k = torch.matmul(P, out_view)
			outPout = torch.matmul(out_view.T, k).item()
			c = 1 / (1 + outPout)
			P = P - c * torch.matmul(k, k.T)
			model.get_layer().forward_weights.data -= c*torch.outer(error.view(-1), k.view(-1))
		
		# compute and print loss
		loss = loss_function(y_pred, data)
		p_bar.set_description(f"Loss: {loss.item():.4f}")


if __name__ == '__main__':
	curbd_data = np.load("data/ts/curbd_Adata.npy")
	network = nt.SequentialModel(
		layers=[nt.WilsonCowanLayer(curbd_data.shape[0], curbd_data.shape[0], activation="tanh", tau=0.1)],
		foresight_time_steps=1,
		out_memory_size=1,
		hh_memory_size=1,
		device=torch.device("cpu"),
	)
	curbd_train(curbd_data, network)



