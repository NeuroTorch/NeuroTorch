from collections import defaultdict
from typing import Optional, Sequence, Union, Dict, Callable, List, Tuple

import numpy as np
import torch

from .learning_algorithm import LearningAlgorithm
from ..learning_algorithms.tbptt import TBPTT
from ..transforms.base import ToDevice
from ..utils import compute_jacobian, list_insert_replace_at


class WeakRLS(TBPTT):
	r"""
	Apply the backpropagation through time algorithm to the given model.
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
		self.K = None
		self.P = None
		self.delta = kwargs.get("delta", 1.0)
		self.lambda_0 = kwargs.get("lambda_0", 0.99)
		self.Lambda = kwargs.get("Lambda", 0.94)
		self.a = kwargs.get("a", 1e-2)
		self.b = kwargs.get("b", 0.9)
		self.alpha = 0.0
		self.Delta = torch.tensor(0.0)
		self._device = kwargs.get("device", None)
		self.to_cpu_transform = ToDevice(device=torch.device("cpu"))
		self.to_device_transform = None
		self.reduction = kwargs.get("reduction", "mean").lower()
		self._other_dims_as_batch = kwargs.get("other_dims_as_batch", False)
		self._is_recurrent = is_recurrent
		self.jacobian_noise = kwargs.get("jacobian_noise", 0.0)
		
		self._asserts()
	
	@property
	def eta(self):
		return max(1e-2, self.alpha * (1 - self.alpha))
	
	def _asserts(self):
		assert self.a < self.b, "a must be less than b"
		assert 0.0 < self.lambda_0 < 1.0, "lambda_0 must be between 0 and 1"
		assert 0.0 < self.Lambda < 1.0, "Lambda must be between 0 and 1"
		assert 0.0 <= self.a <= 1.0, "a must be between 0 and 1"
		assert 0.0 <= self.b <= 1.0, "b must be between 0 and 1"
		assert self.reduction in ["mean", "sum", "none"], "reduction must be either 'mean', 'sum' or 'none'"
	
	def _initialize_K(self, m):
		self.K = [
			torch.zeros((param.numel(), m), dtype=torch.float32, device=torch.device("cpu")).T
			for param in self.params
		]
	
	def _initialize_P(self, m=None):
		self.P = [
			self.delta * torch.eye(param.numel() if m is None else m, dtype=torch.float32, device=torch.device("cpu"))
			for param in self.params
		]
	
	def _update_alpha(self):
		self.alpha = np.clip(self.alpha + self.a, 0.0, self.b)
	
	def _update_delta(self, error):
		self.Delta = self.eta * error + self.alpha * self.Delta
		# self.Delta = 0.1 * error + 0.9 * self.Delta
		# self.Delta = self.alpha * error + self.eta * self.Delta
	
	def _update_lambda(self):
		self.Lambda = self.lambda_0 * self.Lambda + self.lambda_0 * (1 - self.lambda_0)
	
	def _update_k(self, psi):
		# P_psi_list = [
		# 	torch.einsum('ll,lm->lm', self.P[i], psi[i])
		# 	for i in range(len(self.params))
		# ]
		# A_list = [
		# 	(self.Lambda + psi[i].T @ P_psi_list[i]).T
		# 	for i in range(len(self.params))
		# ]
		# rank_list = [torch.linalg.matrix_rank(A) for A in A_list]
		# self.K = [
		# 	# P_psi_list[i] / (self.Lambda + psi[i].T @ P_psi_list[i])
		# 	# P_psi_list[i] / A_list[i]
		# 	# torch.linalg.solve((self.Lambda + psi[i].T @ P_psi_list[i]).T, P_psi_list[i].T).T
		# 	torch.linalg.lstsq(A_list[i], P_psi_list[i].T).solution.T
		# 	# P_psi_list[i] @ torch.linalg.inv(self.Lambda + psi[i].T @ P_psi_list[i])
		# 	for i in range(len(self.params))
		# ]
		self.K = [self.P[i] @ psi[i] for i in range(len(self.params))]
	
	def _update_p(self, psi):
		eyes = [
			self.to_device_transform(torch.eye(self.P[i].shape[0]))
			for i in range(len(self.params))
		]
		self.P = [
			(eyes[i] - self.K[i] @ psi[i].T) @ self.P[i] / self.Lambda
			for i in range(len(self.params))
		]
		
	def _get_psi(self, outputs):
		# TODO: check https://medium.com/@monadsblog/pytorch-backward-function-e5e2b7e60140
		# TODO: see https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py
		# return [self.to_device_transform(param.grad.view(-1)) for param in self.params]
		psi = [[] for _ in range(len(list(self.params)))]
		grad_outputs = torch.eye(outputs.shape[-1])
		for i in range(outputs.shape[-1]):
			self.zero_grad()
			outputs.backward(grad_outputs[i], retain_graph=True)
			for p_idx, param in enumerate(self.params):
				grad = param.grad.flatten().detach().clone()
				grad_noise = torch.randn_like(grad) * self.jacobian_noise
				psi[p_idx].append(grad + grad_noise)
		psi = [torch.stack(psi[i], dim=-1).T for i in range(len(list(self.params)))]
		# psi = compute_jacobian(params=self.params, y=outputs, strategy="slow")
		# psi = [param.grad.view(-1, 1).detach().clone() for param in self.params]
		return psi
	
	def _get_psi_batch(self, batch_outputs):
		psi = [[] for _ in range(len(self.params))]
		for outputs in batch_outputs:
			psi_batch = self._get_psi(outputs)
			for i in range(len(self.params)):
				psi[i].append(psi_batch[i])
		psi = [torch.stack(psi[i], dim=0) for i in range(len(self.params))]
		return psi
	
	def _update_params(self):
		for param, k in zip(self.params, self.K):
			param.data += (k.T @ self.Delta.view(-1, 1)).to(param.device, non_blocking=True).view(param.data.shape)
		# for param, k in zip(self.params, self.K):
		# 	param.grad = -(k.T @ self.Delta.view(-1, 1)).to(param.device, non_blocking=True).view(param.data.shape)
		# self.optimizer.step()
	
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
			if len(self._layers_buffer[layer_name]) == self.backward_time_steps and t != 0:
				# self._backward_at_t(t, self.backward_time_steps, layer_name)
				self._backward_at_t(t, len(self._layers_buffer[layer_name]), layer_name)
				out = self._detach_out(out)
			return out
		
		return _forward
	
	def _backward_at_t(self, t: int, backward_t: int, layer_name: str):
		y_batch = self._get_y_batch_slice_from_trainer((t + 1) - backward_t, t + 1, layer_name)
		pred_batch = self._get_pred_batch_from_buffer(layer_name)
		self._batch_step(pred_batch, y_batch)
		self._layers_buffer[layer_name].clear()
	
	def load_checkpoint_state(self, trainer, checkpoint: dict, **kwargs):
		if self.save_state:
			state = checkpoint.get(self.name, {})
			
	def get_checkpoint_state(self, trainer, **kwargs) -> object:
		if self.save_state:
			return {
			
			}
		return None
	
	def start(self, trainer, **kwargs):
		LearningAlgorithm.start(self, trainer, **kwargs)
		if not self.params:
			self.params = list(trainer.model.parameters())
			
		# filter params to get only the ones that require gradients
		self.params = [param for param in self.params if param.requires_grad]
		self.optimizer = torch.optim.SGD(self.params, lr=1e-2)
		
		if self.criterion is None and trainer.criterion is not None:
			self.criterion = trainer.criterion
		
		if self._device is None:
			self._device = trainer.model.device
		self.to_device_transform = ToDevice(device=self._device)
		# self._initialize_P()
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
	
	def apply_criterion(self, pred_batch, y_batch, criterion):
		if criterion is None:
			if isinstance(y_batch, dict):
				criterion = {key: torch.nn.MSELoss() for key in y_batch}
			else:
				criterion = torch.nn.MSELoss()
		
		if isinstance(criterion, dict):
			raise NotImplementedError("y_batch as dict is not implemented yet")
			if isinstance(y_batch, torch.Tensor):
				y_batch = {k: y_batch for k in criterion}
			if isinstance(pred_batch, torch.Tensor):
				pred_batch = {k: pred_batch for k in criterion}
			assert isinstance(pred_batch, dict) and isinstance(y_batch, dict), \
				"If criterion is a dict, pred, y_batch and pred must be a dict too."
			batch_loss = sum(
				[
					criterion[k](pred_batch[k], y_batch[k].to(pred_batch[k].device))
					for k in criterion
				]
			)
		else:
			if isinstance(pred_batch, dict) and len(pred_batch) == 1:
				pred_batch = pred_batch[list(pred_batch.keys())[0]]
			batch_loss = criterion(pred_batch, y_batch.to(pred_batch.device))
		return batch_loss
	
	def zero_grad(self):
		for param in self.params:
			if param.grad is not None:
				param.grad.detach_()
				param.grad.zero_()
	
	def _update_k_p_on_datum(self, psi, error):
		self._update_k(psi)
		self._update_p(psi)
		
	def _batch_step(self, pred_batch, y_batch):
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
			single_psi = [psi_i[idx] for psi_i in psi]
			# self._update_k_p_on_datum(psi=single_psi, error=error[idx])
			K = [self.P[i] @ single_psi[i] for i in range(len(self.params))]
			self.P = [
				(eyes[i] - K[i] @ single_psi[i].T) @ self.P[i] / self.Lambda
				for i in range(len(self.params))
			]
			lr = 0.1 * error_i
			for param, k in zip(self.params, self.K):
				param.grad = (
						k.T @ lr.reshape(-1, 1)
				).to(param.device, non_blocking=True).reshape(param.data.shape).clone()  # .T?
			# self.K = [self.P[i] @ single_psi[i] for i in range(len(self.params))]
			# psiPpsi = [single_psi[i].T @ K[i] for i in range(len(self.params))]
			# c = [1 / (1 + psiPpsi[i]) for i in range(len(self.params))]
			# self.P = [
			# 	self.P[i] - c[i] * (self.K[i] @ self.K[i].T)
			# 	for i in range(len(self.params))
			# ]
		# self._update_delta(error.mean(dim=0))
		# self._update_params()
		self.optimizer.step()
		self._put_on_cpu()
		self.trainer.model.to(model_device, non_blocking=True)
	
	def on_iteration_begin(self, trainer, **kwargs):
		# self._update_lambda()
		# self._update_alpha()
		pass
	
	def _try_put_on_device(self, trainer):
		try:
			self.K = [self.to_device_transform(k) for k in self.K]
			self.P = [self.to_device_transform(p) for p in self.P]
			self.Delta = self.to_device_transform(self.Delta)
		except Exception as e:
			trainer.model = self.to_cpu_transform(trainer.model)
			self.K = [self.to_device_transform(k) for k in self.K]
			self.P = [self.to_device_transform(p) for p in self.P]
			self.Delta = self.to_device_transform(self.Delta)
			
	def _put_on_cpu(self):
		self.K = [self.to_cpu_transform(k) for k in self.K]
		self.P = [self.to_cpu_transform(p) for p in self.P]
		self.Delta = self.to_cpu_transform(self.Delta)
	
	def on_optimization_begin(self, trainer, **kwargs):
		y_batch = trainer.current_training_state.y_batch
		pred_batch = trainer.format_pred_batch(trainer.current_training_state.pred_batch, y_batch)
		
		if self._is_recurrent:
			for layer_name in self._layers_buffer:
				backward_t = len(self._layers_buffer[layer_name])
				if backward_t > 0:
					self._backward_at_t(self._data_n_time_steps - 1, backward_t, layer_name)
		else:
			self._batch_step(trainer, **kwargs)
		
		trainer.update_state_(batch_loss=self.apply_criterion(pred_batch, y_batch, self.eval_criterion).detach_())
	
	def on_optimization_end(self, trainer, **kwargs):
		self.zero_grad()
		# eval_loss = self.compute_eval_loss(trainer, **kwargs)
		# trainer.update_itr_metrics_state_(eval_criterion=eval_loss)
		metrics = dict(
			psi_mean=self.to_cpu_transform(torch.cat([p.flatten() for p in self.psi])).mean(),
			K_mean=self.to_cpu_transform(torch.cat([k.flatten() for k in self.K])).mean(),
			P_mean=self.to_cpu_transform(torch.cat([p.flatten() for p in self.P])).mean(),
			P_std=self.to_cpu_transform(torch.cat([p.flatten() for p in self.P])).std(),
			Delta_mean=self.to_cpu_transform(self.Delta).mean(),
			# alpha=self.alpha,
			# eta=self.eta,
			# lbda=self.Lambda,
		)
		trainer.update_itr_metrics_state_(**metrics)
	
	def compute_eval_loss(self, trainer, **kwargs):
		y_batch = trainer.current_training_state.y_batch
		pred_batch = trainer.format_pred_batch(trainer.current_training_state.pred_batch, y_batch)
		with torch.no_grad():
			eval_loss = self.apply_criterion(pred_batch, y_batch, self.eval_criterion)
		return eval_loss
	
	def compute_loss(self, trainer, **kwargs):
		y_batch = trainer.current_training_state.y_batch
		pred_batch = trainer.format_pred_batch(trainer.current_training_state.pred_batch, y_batch)
		loss = self.apply_criterion(pred_batch, y_batch, self.criterion)
		return loss
	
	def on_validation_batch_begin(self, trainer, **kwargs):
		eval_loss = self.compute_eval_loss(trainer, **kwargs)
		batch_loss = self.compute_loss(trainer, **kwargs)
		trainer.update_state_(batch_loss=batch_loss)
		trainer.update_itr_metrics_state_(eval_loss=eval_loss)
	
	def on_pbar_update(self, trainer, **kwargs) -> dict:
		return {
			# "eval_loss": self.compute_eval_loss(trainer, **kwargs),
		}

