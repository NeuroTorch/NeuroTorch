from typing import Optional, Sequence, Union, Dict, Callable, List

import numpy as np
import torch

from .learning_algorithm import LearningAlgorithm
from ..transforms.base import ToDevice
from ..utils import compute_jacobian


class WeakRLS(LearningAlgorithm):
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
			**kwargs
	):
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
		super().__init__(**kwargs)
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
		self._asserts()
	
	@property
	def eta(self):
		return self.alpha * (1 - self.alpha)
	
	def _asserts(self):
		assert self.a < self.b, "a must be less than b"
		assert 0.0 < self.lambda_0 < 1.0, "lambda_0 must be between 0 and 1"
		assert 0.0 < self.Lambda < 1.0, "Lambda must be between 0 and 1"
		assert 0.0 <= self.a <= 1.0, "a must be between 0 and 1"
		assert 0.0 <= self.b <= 1.0, "b must be between 0 and 1"
		assert self.reduction in ["mean", "sum", "none"], "reduction must be either 'mean', 'sum' or 'none'"
	
	def _initialize_K(self, m):
		self.K = [
			torch.zeros((param.numel(), m), dtype=torch.float32, device=torch.device("cpu"))
			for param in self.params
		]
	
	def _initialize_P(self):
		self.P = [
			self.delta * torch.eye(param.numel(), dtype=torch.float32, device=torch.device("cpu"))
			for param in self.params
		]
	
	def _update_alpha(self):
		self.alpha = np.clip(self.alpha + self.a, 0.0, self.b)
	
	def _update_delta(self, error):
		self.Delta = self.eta * error + self.alpha * self.Delta
	
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
			self.to_device_transform(torch.eye(self.params[i].numel()))
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
		for output in outputs:
			self.zero_grad()
			output.backward(retain_graph=True)
			for i, param in enumerate(self.params):
				psi[i].append(param.grad.view(-1).detach().clone())
		psi = [torch.stack(psi[i], dim=-1) for i in range(len(list(self.params)))]
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
	
	def _update_theta(self):
		for param, k in zip(self.params, self.K):
			param.data += (k @ self.Delta.view(-1, 1)).to(param.device, non_blocking=True).view(param.data.shape)
	
	def load_checkpoint_state(self, trainer, checkpoint: dict, **kwargs):
		if self.save_state:
			state = checkpoint.get(self.name, {})
			
	def get_checkpoint_state(self, trainer, **kwargs) -> object:
		if self.save_state:
			return {
			
			}
		return None
	
	def start(self, trainer, **kwargs):
		super().start(trainer)
		if not self.params:
			self.params = list(trainer.model.parameters())
		
		if self.criterion is None and trainer.criterion is not None:
			self.criterion = trainer.criterion
		
		if self._device is None:
			self._device = trainer.model.device
		self.to_device_transform = ToDevice(device=self._device)
		self._initialize_P()
	
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
		
	def _batch_step(self, trainer, **kwargs):
		y_batch = trainer.current_training_state.y_batch
		pred_batch = trainer.format_pred_batch(trainer.current_training_state.pred_batch, y_batch)
		assert isinstance(pred_batch, torch.Tensor), "pred_batch must be a torch.Tensor"
		assert isinstance(y_batch, torch.Tensor), "y_batch must be a torch.Tensor"
		
		# pred_batch_view, y_batch_view = pred_batch.view(-1, pred_batch.shape[-1]), y_batch.view(-1, y_batch.shape[-1])
		pred_batch_view, y_batch_view = pred_batch.view(pred_batch.shape[0], -1), y_batch.view(y_batch.shape[0], -1)
		self.zero_grad()
		
		error = self.to_device_transform(y_batch_view - pred_batch_view)
		# mse_loss = torch.nn.MSELoss()(pred_batch_view, y_batch_view)
		# mse_loss.backward()
		# error = mse_loss.unsqueeze(0)
		
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
		self._update_lambda()
		for idx, error_i in enumerate(error):
			single_psi = [psi_i[idx] for psi_i in psi]
			self._update_k_p_on_datum(psi=single_psi, error=error[idx])
		self._update_alpha()
		self._update_delta(error.mean(dim=0))
		self._update_theta()
	
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
		model_device = trainer.model.device
		
		if self.K is None:
			self._initialize_K(m=y_batch.shape[-1])
			# self._initialize_K(m=1)
		
		self._try_put_on_device(trainer)
		self._batch_step(trainer, **kwargs)
		self._put_on_cpu()
		trainer.model.to(model_device, non_blocking=True)
		
		trainer.update_state_(batch_loss=self.apply_criterion(pred_batch, y_batch, self.eval_criterion).detach_())
	
	def on_optimization_end(self, trainer, **kwargs):
		self.zero_grad()
		# eval_loss = self.compute_eval_loss(trainer, **kwargs)
		# trainer.update_itr_metrics_state_(eval_criterion=eval_loss)
	
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

