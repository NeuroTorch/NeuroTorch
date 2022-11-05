from collections import defaultdict
from typing import Optional, Sequence, Union, Dict, Callable, List, Tuple

import numpy as np
import torch

from .learning_algorithm import LearningAlgorithm
from ..learning_algorithms.tbptt import TBPTT
from ..transforms.base import ToDevice
from ..utils import compute_jacobian, list_insert_replace_at


class RLS(TBPTT):
	r"""
	Apply the recursive least squares algorithm to the given model.
	"""
	CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: str = "optimizer_state_dict"
	
	def __init__(
			self,
			*,
			params: Optional[Sequence[torch.nn.Parameter]] = None,
			layers: Optional[Union[Sequence[torch.nn.Module], torch.nn.Module]] = None,
			criterion: Optional[Union[Dict[str, Union[torch.nn.Module, Callable]], torch.nn.Module, Callable]] = None,
			backward_time_steps: Optional[int] = None,
			is_recurrent: bool = False,
			**kwargs
	):
		kwargs.setdefault("auto_backward_time_steps_ratio", 0)
		"""
		Constructor for WeakRLS class.

		:param params: The parameters to optimize. If None, the parameters of the model's trainer will be used.
		:type params: Optional[Sequence[torch.nn.Parameter]]
		:param criterion: The criterion to use. If not provided, torch.nn.MSELoss is used.
		:type criterion: Optional[Union[Dict[str, Union[torch.nn.Module, Callable]], torch.nn.Module, Callable]]
		:param kwargs: The keyword arguments to pass to the BaseCallback.

		:keyword bool save_state: Whether to save the state of the optimizer. Defaults to True.
		:keyword bool load_state: Whether to load the state of the optimizer. Defaults to True.
		"""
		kwargs.setdefault("save_state", True)
		kwargs.setdefault("load_state", True)
		super().__init__(
			params=params,
			layers=layers,
			criterion=criterion,
			backward_time_steps=backward_time_steps,
			optimizer=None,
			optim_time_steps=None,
			**kwargs
		)
		if params is None:
			params = []
		else:
			params = list(params)
		if layers is not None:
			if isinstance(layers, torch.nn.Module):
				layers = [layers]
			params.extend([param for layer in layers for param in layer.parameters() if param not in params])
		self.params: List[torch.nn.Parameter] = params
		self.layers = layers
		self.eval_criterion = criterion
		self.criterion = torch.nn.MSELoss()
		
		# RLS attributes
		self.P_list = None
		self.delta = kwargs.get("delta", 1.0)
		self.Lambda = kwargs.get("Lambda", 1.0)
		self._device = kwargs.get("device", None)
		self.to_cpu_transform = ToDevice(device=torch.device("cpu"))
		self.to_device_transform = None
		self._other_dims_as_batch = kwargs.get("other_dims_as_batch", False)
		self._is_recurrent = is_recurrent
		self.step_mth = kwargs.get("step_mth", "inputs")
		self.kwargs = kwargs
		self._asserts()
	
	def _asserts(self):
		assert 0.0 < self.Lambda <= 1.0, "Lambda must be between 0 and 1"
	
	def initialize_P_list(self, m=None):
		self.P_list = [
			self.delta * torch.eye(param.numel() if m is None else m, dtype=torch.float32, device=torch.device("cpu"))
			for param in self.params
		]
	
	def _maybe_update_time_steps(self):
		if self._auto_set_backward_time_steps:
			self.backward_time_steps = max(1, int(self._auto_backward_time_steps_ratio * self._data_n_time_steps))
	
	def _decorate_forward(self, forward, layer_name: str):
		def _forward(*args, **kwargs):
			out = forward(*args, **kwargs)
			t = kwargs.get("t", None)
			if t is None:
				return out
			out_tensor = self._get_out_tensor(out)
			list_insert_replace_at(self._layers_buffer[layer_name], t % self.backward_time_steps, out_tensor)
			if len(self._layers_buffer[layer_name]) == self.backward_time_steps:
				self._backward_at_t(t, self.backward_time_steps, layer_name)
				out = self._detach_out(out)
			return out
		
		return _forward
	
	def _backward_at_t(self, t: int, backward_t: int, layer_name: str):
		x_batch = self._get_x_batch(t, backward_t)  # TODO: get x_batch from buffer
		y_batch = self._get_y_batch_slice_from_trainer((t + 1) - backward_t, t + 1, layer_name)
		pred_batch = self._get_pred_batch_from_buffer(layer_name)
		self.optimization_step(x_batch, pred_batch, y_batch)
		self._layers_buffer[layer_name].clear()
	
	def load_checkpoint_state(self, trainer, checkpoint: dict, **kwargs):
		if self.save_state:
			state = checkpoint.get(self.name, {})
			
	def get_checkpoint_state(self, trainer, **kwargs) -> object:
		if self.save_state:
			return {
			
			}
		return None
	
	def _try_put_on_device(self, trainer):
		try:
			self.P = [self.to_device_transform(p) for p in self.P]
		except Exception as e:
			trainer.model = self.to_cpu_transform(trainer.model)
			self.P = [self.to_device_transform(p) for p in self.P]
	
	def _put_on_cpu(self):
		self.P = [self.to_cpu_transform(p) for p in self.P]
	
	def start(self, trainer, **kwargs):
		LearningAlgorithm.start(self, trainer, **kwargs)
		if self.params and self.optimizer is None:
			self.optimizer = torch.optim.SGD(self.params, lr=self.kwargs.get("lr", 1.0))
		elif not self.params and self.optimizer is not None:
			self.params.extend(
				[
					param
					for i in range(len(self.optimizer.param_groups))
					for param in self.optimizer.param_groups[i]["params"]
				]
			)
		else:
			self.params = trainer.model.parameters()
			self.optimizer = torch.optim.Adam(self.params)
		
		if self.criterion is None and trainer.criterion is not None:
			self.criterion = trainer.criterion
			
		# filter params to get only the ones that require gradients
		self.params = [param for param in self.params if param.requires_grad]
		
		if self._device is None:
			self._device = trainer.model.device
		self.to_device_transform = ToDevice(device=self._device)
		self.output_layers: torch.nn.ModuleDict = trainer.model.output_layers
		self._initialize_original_forwards()
	
	def on_batch_begin(self, trainer, **kwargs):
		LearningAlgorithm.on_batch_begin(self, trainer, **kwargs)
		self.trainer = trainer
		if self._is_recurrent:
			self._data_n_time_steps = self._get_data_time_steps_from_y_batch(trainer.current_training_state.y_batch)
			self._maybe_update_time_steps()
			self.decorate_forwards()
	
	def on_batch_end(self, trainer, **kwargs):
		super().on_batch_end(trainer)
		self.undecorate_forwards()
		self._layers_buffer.clear()
	
	def on_optimization_begin(self, trainer, **kwargs):
		x_batch = trainer.current_training_state.x_batch
		y_batch = trainer.current_training_state.y_batch
		pred_batch = trainer.format_pred_batch(trainer.current_training_state.pred_batch, y_batch)
		
		if self._is_recurrent:
			for layer_name in self._layers_buffer:
				backward_t = len(self._layers_buffer[layer_name])
				if backward_t > 0:
					self._backward_at_t(self._data_n_time_steps - 1, backward_t, layer_name)
		else:
			self.optimization_step(x_batch, pred_batch, y_batch)
		
		trainer.update_state_(batch_loss=self.apply_criterion(pred_batch, y_batch).detach_())
	
	def optimization_step(self, x_batch: torch.Tensor, pred_batch: torch.Tensor, y_batch: torch.Tensor):
		name_to_mth = {
			"inputs": self.inputs_mth_step,
			"outputs": self.outputs_mth_step,
			"grad": self.grad_mth_step,
			"jacobian": self.jacobian_mth_step,
		}
		if self.step_mth not in name_to_mth:
			raise ValueError(f"Invalid step_mth: {self.step_mth}")
		return name_to_mth[self.step_mth](x_batch, pred_batch, y_batch)
	
	def jacobian_mth_step(self, x_batch: torch.Tensor, pred_batch: torch.Tensor, y_batch: torch.Tensor):
		jac = compute_jacobian(params=self.params, y=pred_batch, strategy="slow")
	
	def grad_mth_step(self, x_batch: torch.Tensor, pred_batch: torch.Tensor, y_batch: torch.Tensor):
		model_device = self.trainer.model.device
		assert isinstance(pred_batch, torch.Tensor), "pred_batch must be a torch.Tensor"
		assert isinstance(y_batch, torch.Tensor), "y_batch must be a torch.Tensor"
		
		if self._other_dims_as_batch:
			pred_batch_view, y_batch_view = pred_batch.view(-1, pred_batch.shape[-1]), y_batch.view(
				-1, y_batch.shape[-1]
			)
		else:
			pred_batch_view, y_batch_view = pred_batch.view(pred_batch.shape[0], -1), y_batch.view(y_batch.shape[0], -1)
		
		pred_batch_view, y_batch_view = torch.mean(pred_batch_view, dim=0), torch.mean(y_batch_view, dim=0)
		
		if self.P is None:
			self.P = [
				torch.eye(
					param.numel(), dtype=torch.float32, device=torch.device("cpu")
				)
				for param in self.params
			]
		
		error = self.to_device_transform(pred_batch_view - y_batch_view)
		# K = [torch.matmul(self.P[i], pred_batch_view) for i in range(len(self.params))]
		# yPy = [torch.matmul(pred_batch_view.T, K[i]).item() for i in range(len(self.params))]
		# c = [1.0 / (1.0 + yPy[i]) for i in range(len(self.params))]
		# self.P = [self.P[i] - c[i] * torch.matmul(K[i], K[i].T) for i in range(len(self.params))]
		# delta_w = [c[i] * torch.outer(error.flatten(), K[i].flatten()) for i in range(len(self.params))]
		
		self.optimizer.zero_grad()
		loss = torch.nn.MSELoss()(pred_batch_view, y_batch_view.to(model_device))
		loss.backward()
		psi = [param.grad.detach().view(-1, 1).clone() for param in self.params]
		# self._try_put_on_device(self.trainer)
		self.psi = psi
		K = [torch.matmul(self.P[i], psi[i]) for i in range(len(self.params))]
		gradPgrad = [torch.matmul(psi[i].T, K[i]).item() for i in range(len(self.params))]
		c = [1.0 / (1.0 + gradPgrad[i]) for i in range(len(self.params))]
		self.P = [self.P[i] - c[i] * torch.matmul(K[i], K[i].T) for i in range(len(self.params))]
		delta_w = [c[i] * K[i].view(-1) for i in range(len(self.params))]
		self.mean_delta_w = [torch.mean(delta_w[i]) for i in range(len(self.params))]
		for i, param in enumerate(self.params):
			param.grad = delta_w[i].to(param.device, non_blocking=True).view(param.data.shape).clone()
		self.optimizer.step()
		# self._put_on_cpu()
		self.trainer.model.to(model_device, non_blocking=True)
	
	def inputs_mth_step(self, x_batch: torch.Tensor, pred_batch: torch.Tensor, y_batch: torch.Tensor):
		"""

		x.shape = [B, N_in]
		y.shape = [B, N_out]
		error.shape = [B, N_out]
		epsilon.shape = [1, N_out]
		phi.shape = [1, N_in]

		P.shape = [N_in, N_in]
		K = P[N_in, N_in] @ phi.T[N_in, 1] -> [N_in, 1]
		h = 1 / (labda[1] + kappa[1] * phi[1, N_in] @ K[N_in, 1]) -> [1]
		P = labda[1] * P[N_in, N_in] - h[1] * kappa[1] * K[N_in, 1] @ K.T[1, N_in] -> [N_in, N_in]
		grad = h[1] * K[N_in, 1] @ epsilon[1, N_out] -> [N_in, N_out]

		:param inputs: inputs of the layer
		:param output: outputs of the layer
		:param target: targets of the layer
		:param P: inverse covariance matrix of hte inputs
		:param optimizer: optimizer of the layer
		:param kwargs: Additional parameters

		:return: The updated inverse covariance matrix
		"""
		model_device = self.trainer.model.device
		assert isinstance(x_batch, torch.Tensor), "x_batch must be a torch.Tensor"
		assert isinstance(pred_batch, torch.Tensor), "pred_batch must be a torch.Tensor"
		assert isinstance(y_batch, torch.Tensor), "y_batch must be a torch.Tensor"
		
		labda = self.kwargs.get("labda", 1.0)
		kappa = self.kwargs.get("kappa", 1.0)
		
		x_batch_view = x_batch.view(-1, x_batch.shape[-1])  # [B, f_in]
		pred_batch_view = pred_batch.view(-1, pred_batch.shape[-1])  # [B, f_out]
		y_batch_view = y_batch.view(-1, y_batch.shape[-1])  # [B, f_out]
		error = self.to_device_transform(pred_batch_view - y_batch_view)  # [B, f_out]
		
		if self.P_list is None:
			self.initialize_P_list()
		
		epsilon = error.mean(dim=0).view(1, -1)  # [1, f_out]
		phi = x_batch_view.mean(dim=0).view(1, -1).detach().clone()  # [1, f_in]
		K_list = [torch.matmul(P, phi.T) for P in self.P_list]  # [f_in, f_in] @ [f_in, 1] -> [f_in, 1]
		h = [1.0 / (labda + kappa * torch.matmul(phi, K)).item() for K in K_list]  # [1, f_in] @ [f_in, 1] -> [1]
		
		for p in self.params:
			p.grad = h * torch.outer(K.view(-1), epsilon.view(-1))  # [f_in, 1] @ [1, N_out] -> [N_in, N_out]
		self.optimizer.zero_grad()

		self.optimizer.step()
		P = labda * P - h * kappa * torch.matmul(K, K.T)  # [N_in, 1] @ [1, N_in] -> [N_in, N_in]
		
		self._put_on_cpu()
		self.trainer.model.to(model_device, non_blocking=True)
	
	def _batch_step_subhi(self, pred_batch, y_batch):
		model_device = self.trainer.model.device
		assert isinstance(pred_batch, torch.Tensor), "pred_batch must be a torch.Tensor"
		assert isinstance(y_batch, torch.Tensor), "y_batch must be a torch.Tensor"
		
		if self._other_dims_as_batch:
			pred_batch_view, y_batch_view = pred_batch.view(-1, pred_batch.shape[-1]), y_batch.view(-1, y_batch.shape[-1])
		else:
			pred_batch_view, y_batch_view = pred_batch.view(pred_batch.shape[0], -1), y_batch.view(y_batch.shape[0], -1)
		self.zero_grad()
		
		if self.K is None:
			self._initialize_K(m=y_batch_view.shape[-1])
		if self.P is None:
			self._initialize_P(m=y_batch_view.shape[-1])
		self._try_put_on_device(self.trainer)
		
		error = self.to_device_transform(y_batch_view - pred_batch_view)
		if self.reduction == "mean":
			error = error.mean(dim=0).unsqueeze(0)
			psi = self._get_psi_batch(pred_batch_view.mean(dim=0).unsqueeze(0))
		elif self.reduction == "sum":
			error = error.sum(dim=0).unsqueeze(0)
			psi = self._get_psi_batch(pred_batch_view.sum(dim=0).unsqueeze(0))
		elif self.reduction == "none":
			psi = self._get_psi_batch(pred_batch_view)
		else:
			raise ValueError(f"reduction must be one of 'mean', 'sum', 'none', got {self.reduction}")
		
		psi = self.to_device_transform(psi)
		self.psi = psi
		
		eyes = [
			self.to_device_transform(torch.eye(self.P[i].shape[0]))
			for i in range(len(self.params))
		]
		self.optimizer.zero_grad()
		for idx, error_i in enumerate(error):
			single_psi = [10*psi_i[idx] for psi_i in psi]
			# self._update_k_p_on_datum(psi=single_psi, error=error[idx])
			K = [self.P[i] @ single_psi[i] for i in range(len(self.params))]
			# self.P = [
			# 	(eyes[i] - K[i] @ single_psi[i].T) @ self.P[i] / self.Lambda
			# 	for i in range(len(self.params))
			# ]
			self.P = [self.P[i] - torch.matmul(K[i], K[i].T) / self.Lambda for i in range(len(self.params))]
			delta_w = [torch.matmul(error_i.view(1, -1), K[i]) for i in range(len(self.params))]
			self.mean_delta_w = [torch.mean(delta_w[i]) for i in range(len(self.params))]
			for i, param in enumerate(self.params):
				param.data += delta_w[i].to(param.device, non_blocking=True).view(param.data.shape).clone()  # .T?
		# self._update_delta(error.mean(dim=0))
		# self._update_params()
		# self.optimizer.step()
		self._put_on_cpu()
		self.trainer.model.to(model_device, non_blocking=True)

