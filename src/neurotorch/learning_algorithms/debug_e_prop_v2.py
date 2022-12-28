import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

from neurotorch import to_numpy, to_tensor
from typing import *
from neurotorch.modules import layers
from neurotorch.dimension import SizeTypes
from neurotorch.metrics import PVarianceLoss
from neurotorch.modules.layers import Linear, LILayer, WilsonCowanLayer, SpyLILayer, BellecLIFLayer
from neurotorch.utils import unpack_out_hh, legend_without_duplicate_labels_, batchwise_temporal_recursive_filter


class SimplifiedEprop:

	def __init__(
			self,
			true_time_series: torch.Tensor,
			model: layers.BaseLayer,
			learning_rate: float = 1e-2,
			update_each: int = 1,
			iteration: int = 100,
			device: Optional[torch.device] = torch.device("cpu"),
			**kwargs
	):
		self.true_time_series = to_tensor(true_time_series)
		self.model = model
		self.learning_rate = learning_rate
		self.update_each = update_each
		self.iteration = iteration
		self.device = device
		self.kwargs = kwargs
		self._set_default_kwargs()
		self.out = {}
		self.filtered_eligibility_trace_t = torch.zeros_like(self.model.forward_weights, dtype=self.model.forward_weights.dtype, device=self.model.forward_weights.device)
		self.loss = torch.zeros_like(self.true_time_series)

		self.update_per_iter = math.ceil(true_time_series.shape[0] // self.update_each)

		self.eligibility_trace_t = []
		self.eligibility_trace_t_minus_1 = None
		self.learning_signal_with_eligibility_trace_at_t = []
		self.delta_params = []

	def _set_default_kwargs(self):
		self.kwargs.setdefault("kappa", 0.0)
		random_matrix = torch.randn(self.model.forward_weights.shape, dtype=self.model.forward_weights.dtype, device=self.model.forward_weights.device)
		self.kwargs.setdefault("B", random_matrix)

	def begin(self):
		self.model.build()
		for param in self.model.parameters():
			self.delta_params.append(torch.zeros_like(param, dtype=param.dtype, device=param.device))
			self.eligibility_trace_t.append(torch.zeros_like(param, dtype=param.dtype, device=param.device))
			self.learning_signal_with_eligibility_trace_at_t.append(torch.zeros_like(param, dtype=param.dtype, device=param.device))
		with torch.no_grad():
			self.out["true_time_series"] = self.true_time_series
			self.out["W0"] = self.model.forward_weights.clone()
			self.out["mu0"] = self.model.mu.clone()
			self.out["r0"] = self.model.r.clone()
			self.out["tau0"] = self.model.tau.clone()
			self.out["B"] = self.kwargs["B"].clone()
			self.out["kappa"] = self.kwargs["kappa"].clone()
			return self

	def train(self):
		self.begin()
		progress_bar = tqdm(
			range(self.iteration),
			total=self.iteration,
			desc="Training",
			unit="iteration",
		)

		for _ in progress_bar:
			x_pred = []
			x_pred.append(self.true_time_series[0].clone())
			forward_tensor = self.true_time_series[0].clone()

			for t in range(1, self.true_time_series.shape[0]):
				forward_tensor = self.model.forward(forward_tensor)
				x_pred.append(forward_tensor)
				self.compute_dz_dw_local(forward_tensor)




			# for update_complete in range(self.update_per_iter):
			# 	x_pred = []
			# 	x_pred.append(self.true_time_series[0].clone())
			# 	forward_tensor = self.true_time_series[0].clone()
			#
			# 	for i in range(self.update_each)
			# 		forward_tensor = self.model(forward_tensor)
			# 		x_pred.append(forward_tensor)


	def update_parameters(self):
		pass


	def compute_dz_dw_local(self, z: torch.tensor):
		"""
		Equation (13)
		"""
		for param_idx, param in enumerate(self.model.parameters()):
			# for each neuron at a time step t
			for neuron_idx in range(self.true_time_series.shape[1]):
				if param.requires_grad:
					self.eligibility_trace_t[param_idx][:, neuron_idx] = torch.autograd.grad(z[neuron_idx], param, retain_graph=True)[0][:, neuron_idx]


	def compute_learning_signal_with_eligibility_trace(self, loss_at_t: torch.tensor):
		"""
		Compute L_j^t @ e_{ij}^t for equation (28)
		Since loss = y^t - y_pred^t where y [1xN]
		loss @ B.T = B @ loss.T -> loss.T is the biological convention where loss is the NeuroTorch convention
		"""
		# self.filtered_eligibility_trace_t()
		learning_signal_at_t = loss_at_t @ self.kwargs["B"].T
		for param_idx, _ in enumerate(self.model.parameters()):
			self.learning_signal_with_eligibility_trace_at_t = learning_signal_at_t * self.eligibility_trace_t[param_idx]




	def filter_eligibility_trace_t(self):
		"""
		TODO : Equation (12)
		"""
		pass

	def optimize(self):
		"""
		Equation (28)
		"""
		pass