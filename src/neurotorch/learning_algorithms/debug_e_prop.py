"""
Goal: Make e-prop with auto-differentiation.
References paper:
	- https://arxiv.org/pdf/2201.07602.pdf
	- RFLO: https://elifesciences.org/articles/43299#s4
	- Bellec: https://www.biorxiv.org/content/10.1101/738385v3.full.pdf+html
	
References code:
	- https://github.com/ChFrenkel/eprop-PyTorch/blob/main/models.py
	- Bellec: https://github.com/IGITUGraz/eligibility_propagation
"""
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from neurotorch import to_numpy, to_tensor
from neurotorch.dimension import SizeTypes
from neurotorch.metrics import PVarianceLoss
from neurotorch.modules.layers import Linear, LILayer, WilsonCowanLayer, SpyLILayer, BellecLIFLayer
from neurotorch.utils import unpack_out_hh, legend_without_duplicate_labels_, batchwise_temporal_recursive_filter


def compute_dz_dw_local(z, w):
	z_view = z.view(-1)
	dzdw = torch.zeros_like(w)
	grad_outputs = torch.eye(z_view.shape[-1], device=w.device)
	for g_idx in range(grad_outputs.shape[0]):
		w.grad = torch.zeros_like(w)
		dzdw[g_idx] = torch.autograd.grad(z_view[g_idx], w, retain_graph=True)[0][g_idx]
	return dzdw


class DummyLayer(Linear):
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			device: Optional[torch.device] = None,
			**kwargs
	):
		super().__init__(
			input_size=input_size,
			output_size=output_size,
			name=name,
			device=device,
			**kwargs
		)
		

def dummy_train(targets: torch.Tensor):
	targets = to_tensor(targets)[None, ...]
	snn_layer = BellecLIFLayer(
		input_size=targets.shape[-1],
		output_size=targets.shape[-1],
		use_recurrent_connection=True,
		dt=1.0,
		# hh_init="inputs",
		device=targets.device,
	).build()
	output_layer = LILayer(
		input_size=targets.shape[-1],
		output_size=targets.shape[-1],
		device=targets.device,
		# hh_init="inputs",
		use_bias=False,
		dt=1.0,
	).build()
	feedback_weights = torch.randn((snn_layer.recurrent_weights.shape[0], output_layer.forward_weights.shape[0]))
	optimizer = torch.optim.SGD(output_layer.parameters(), lr=0.1, maximize=True, weight_decay=0.0)
	criterion = PVarianceLoss()
	preds = None
	grad_list = []
	p_bar = tqdm(range(1000))
	for i in p_bar:
		eligibility_trace = []
		snn_out_list, snn_hh_list = [targets[:, 0]], [None]
		output_list, hh_list = [targets[:, 0]], [None]
		for t in range(1, targets.shape[1]):
			snn_out, snn_hh = unpack_out_hh(snn_layer(targets[:, t - 1], snn_hh_list[t-1]))
			snn_hh_list.append(snn_hh)
			snn_out_list.append(snn_out)
			out, hh = unpack_out_hh(output_layer(snn_out_list[t], hh_list[t-1]))
			output_list.append(out)
			hh_list.append(hh)
			
			# Compute eligibility trace
			# instantaneous_eligibility_trace = compute_dz_dw_local()
			# out.detach_()
			# eligibility_trace.append(instantaneous_eligibility_trace)
			
		preds = torch.stack(output_list, dim=1)
		output_error = targets[:, 1:] - preds[:, 1:]
		learning_signals = torch.einsum("btf,fn->btn", output_error, feedback_weights)
		# eligibility_trace = torch.stack(eligibility_trace, dim=0)
		v_scaled = (torch.stack([h[0] for h in snn_hh_list[1:]], dim=1) - snn_layer.threshold) / snn_layer.threshold
		post_term = snn_layer.spike_func.pseudo_derivative(v_scaled, snn_layer.threshold, snn_layer.gamma)
		input_spikes = torch.stack(snn_out_list[1:], dim=1)
		z_previous_time = torch.stack(snn_out_list[1:], dim=1)
		z = torch.stack(snn_out_list[1:], dim=1)
		pre_term_w_in = batchwise_temporal_recursive_filter(input_spikes, decay=0.9)
		pre_term_w_rec = batchwise_temporal_recursive_filter(z_previous_time, decay=0.9)
		pre_term_w_out = batchwise_temporal_recursive_filter(z, decay=0.9)
		eligibility_traces_w_in = post_term[:, :, None, :] * pre_term_w_in[:, :, :, None]
		eligibility_traces_w_rec = post_term[:, :, None, :] * pre_term_w_rec[:, :, :, None]
		
		# To define the gradient of the readout error,
		# the eligibility traces are smoothed with the same filter as the readout
		eligibility_traces_convolved_w_in = batchwise_temporal_recursive_filter(eligibility_traces_w_in, decay=0.9)
		eligibility_traces_convolved_w_rec = batchwise_temporal_recursive_filter(eligibility_traces_w_rec, decay=0.9)
		
		# To define the gradient of the regularization error defined on the averaged firing rate,
		# the eligibility traces should be averaged over time
		eligibility_traces_averaged_w_in = eligibility_traces_w_in.mean(dim=(0, 1))
		eligibility_traces_averaged_w_rec = eligibility_traces_w_rec.mean(dim=(0, 1))
		
		# gradients of the main loss with respect to the weights
		dloss_dw_out = torch.einsum("btno->no", output_error[:, :, None, :] * pre_term_w_out[:, :, :, None])
		dloss_dw_in = torch.einsum("btno->no", learning_signals[:, :, None, :] * eligibility_traces_convolved_w_in)
		dloss_dw_rec = torch.einsum("btno->no", learning_signals[:, :, None, :] * eligibility_traces_convolved_w_rec)
		
		# grad = torch.einsum("tno->no", learning_signal[:, np.newaxis] * eligibility_trace)
		snn_layer.forward_weights.grad = dloss_dw_in
		snn_layer.recurrent_weights.grad = dloss_dw_rec
		output_layer.forward_weights.grad = dloss_dw_out
		
		# optimizer.zero_grad()
		loss = criterion(preds, targets)
		# loss.backward()
		optimizer.step()
		
		mean_grad = np.mean(np.abs(to_numpy(snn_layer.forward_weights.grad)))
		grad_list.append(mean_grad)
		p_bar.set_description(f"Loss: {loss.item():.4f}, mean_grad: {mean_grad:.4f}")
	return torch.squeeze(preds), grad_list


if __name__ == '__main__':
	T = 100
	inputs_spikes = torch.rand((1, T, 1)) < 0.1
	# target = torch.stack([torch.cos(torch.linspace(0, 2 * np.pi, 100)) for i in range(100)], dim=-1)
	# target = (target - target.mean(dim=0, keepdim=True)) / target.std(dim=0, keepdim=True)
	# target = (target - target.min(dim=0, keepdim=True)[0]) / (target.max(dim=0, keepdim=True)[0] - target.min(dim=0, keepdim=True)[0])
	target = torch.stack([-torch.exp(torch.linspace(0, 1, T)) for i in range(1)], dim=-1)
	target = (target - target.min(dim=0, keepdim=True)[0]) / (
				target.max(dim=0, keepdim=True)[0] - target.min(dim=0, keepdim=True)[0])
	if target.ndim == 1:
		target = target[:, None]
	predictions, gradients = dummy_train(target)
	
	fig, axes = plt.subplots(1, 2, figsize=(16, 8))
	axes[0].plot(to_numpy(target)[:, 0], 'o', label="target")
	axes[0].plot(to_numpy(predictions), c='k', label="predictions")
	axes[0].legend()
	legend_without_duplicate_labels_(axes[0])
	
	axes[1].plot(gradients, label="mean abs. gradients")
	axes[1].legend()
	plt.show()

