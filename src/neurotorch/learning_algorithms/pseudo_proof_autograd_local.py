from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

import neurotorch as nt
from neurotorch import to_numpy, to_tensor
from typing import *
from neurotorch.modules import layers
from neurotorch.dimension import SizeTypes
from neurotorch.metrics import PVarianceLoss
from neurotorch.modules.layers import Linear, LILayer, WilsonCowanLayer, SpyLILayer, BellecLIFLayer, WilsonCowanLayerDebug
from neurotorch.utils import unpack_out_hh, legend_without_duplicate_labels_, batchwise_temporal_recursive_filter, \
	filter_parameters, zero_grad_params

"""
Remark: when increasing the batch size, we reduce the performance. Our approximation works fine
if we specify that the batch size is 1. Here's an explanation :
Let say we use a batch of two elements. Let a_1 and a_2 be the gradient dE/dz for each element and b_1 and b_2 the 
gradient for dz/dw for each element. We compute dE/dz and dz/dw independently and sum it up after over the batch size.
The, when we multiply dE/dz with dz/dW element wise, we have (a_1 + a_2) * (b_1 + b_2).
However, what we truly want is (a_1 * b_1 + a_2 * b_2).

The used approximation is the following:
	(a_1 + a_2) * (b_1 + b_2) = (a_1 * b_1 + a_2 * b_2), (sorry Jeff ... :( )

Hence, as B -> inf where B is the batch size, we add useless terms to the gradient which impact our approximation.
This can be manually adjust, but such adjustment really slowed down the computation speed.

TODO: Find an efficient way to compute (a_n * b_n + ... + a_1 * b_1) without using our approximation.
"""


"""
Questions for Bellec:

- Why?
- Who?
- How dare you?
"""


def dE_dw(E, params):
	dE_dw = []
	for param in params:
		if param.requires_grad:
			dE_dw.append(torch.autograd.grad(E, param, retain_graph=True)[0])
	return dE_dw


def dE_dz(E, z):
	return torch.autograd.grad(E, z, retain_graph=True)[0]


def dz_dw_local(z: torch.Tensor, params: Sequence[torch.nn.Parameter]):
	grad_local = []
	for param_idx, param in enumerate(filter_parameters(params, requires_grad=True)):
		grad_local.append(torch.zeros_like(param))
		for unit_idx in range(param.shape[-1]):
			grad_local[param_idx][..., unit_idx] = torch.autograd.grad(
				z[..., unit_idx], param,
				grad_outputs=torch.ones_like(z[..., -1]),
				retain_graph=True,
			)[0][..., unit_idx]
	return grad_local


def dE_dw_local(E, z, w):
	dz_dw_local_list = dz_dw_local(z, w)
	return [torch.sum(dE_dz(E, z), dim=0) * v for v in dz_dw_local_list]


def run_pseudo_proof(layer, input_like, n_tests: int = 1):
	mse = torch.nn.MSELoss()
	# grad_loss = torch.nn.L1Loss()
	grad_loss = nt.losses.PVarianceLoss()
	errors = []
	hh = None
	p_bar = tqdm(range(n_tests))
	for i in p_bar:
		zero_grad_params(layer.parameters())
		inputs = torch.rand_like(input_like)
		z, _ = unpack_out_hh(layer(inputs, hh))
		target = torch.rand_like(z)
		E = mse(z, target)
		true_grad = dE_dw(E, layer.parameters())
		grad_from_local = dE_dw_local(E, z, layer.parameters())
		err = to_numpy(torch.mean(torch.stack([grad_loss(v0, v1) for v0, v1 in zip(grad_from_local, true_grad)])))
		errors.append(err)
		p_bar.set_postfix(err=err)
	return errors


def plot_pseudo_proof(errors):
	mu, sigma = np.mean(errors), np.std(errors)
	n, bins, patches = plt.hist(errors, 50, density=True, facecolor='g', alpha=0.75)

	plt.xlabel('Errors')
	plt.ylabel('Probability')
	plt.title('Histogram of Errors')
	plt.text(mu, 0.5*np.max(n), rf'$\mu={mu:.3e},\ \sigma={sigma:.3e}$')
	# plt.xlim(40, 160)
	# plt.ylim(0, 0.03)
	plt.grid(True)
	plt.show()


if __name__ == '__main__':
	debug_layer = WilsonCowanLayer(
		100, 100,
		# forward_weights=torch.ones(100, 100),
		dt=0.02,
		r=1.0,
		mu=0.5,
		tau=1.0,
		lean_r=False,
		learn_mu=False,
		learn_tau=False,
		hh_init="inputs",
		device=torch.device("cpu"),
		name="WilsonCowanAutogradDebug",
		force_dale_law=False
	).build()

	test_errors = run_pseudo_proof(debug_layer, torch.rand(32, 100), n_tests=1_000)
	plot_pseudo_proof(test_errors)

