import warnings
from typing import Optional, Sequence, Union, Dict, Callable, List

import torch

from .buffers import BatchExperience
from ..learning_algorithms.learning_algorithm import LearningAlgorithm


class PPO(LearningAlgorithm):
	r"""
	Apply the Proximal Policy Optimization algorithm to the given model.
	"""
	CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: str = "optimizer_state_dict"
	
	def __init__(
			self,
			params: Optional[Sequence[torch.nn.Parameter]] = None,
			optimizer: Optional[torch.optim.Optimizer] = None,
			**kwargs
	):
		kwargs.setdefault("save_state", True)
		kwargs.setdefault("load_state", True)
		super().__init__(params=params, **kwargs)
	
	def _compute_continuous_loss(self, batch: BatchExperience, predictions, targets) -> torch.Tensor:
		if torch.numel(batch.continuous_actions) == 0:
			continuous_loss = 0.0
		else:
			targets = (
					batch.rewards
					+ (1.0 - batch.terminals)
					* self.kwargs["gamma"]
					* targets
			).to(self.policy.device)
			continuous_loss = self.continuous_criterion(predictions, targets)
		return continuous_loss

	def _compute_discrete_loss(self, batch: BatchExperience, predictions, targets) -> torch.Tensor:
		if torch.numel(batch.discrete_actions) == 0:
			discrete_loss = 0.0
		else:
			targets = (
					batch.rewards
					+ (1.0 - batch.terminals)
					* self.kwargs["gamma"]
					* targets
			).to(self.policy.device)
			warnings.warn("Discrete loss is not implemented with cross entropy loss. This is a temporary solution.")
			discrete_loss = self.discrete_criterion(predictions, targets)
		return discrete_loss

	def _compute_proximal_policy_optimization_loss(self, batch: BatchExperience, predictions, targets) -> torch.Tensor:
		if torch.numel(batch.continuous_actions) == 0:
			continuous_loss = 0.0
		else:
			targets = (
					batch.rewards
					+ (1.0 - batch.terminals)
					* self.kwargs["gamma"]
					* targets
			).to(self.policy.device)
			continuous_loss = self.continuous_criterion(predictions, targets)
		return continuous_loss
	
	def update_weights(
			self,
			batch: BatchExperience,
			predictions,
			targets,
	) -> float:
		warnings.warn("This method is deprecated. Please use update_policy_weights instead.", DeprecationWarning)
		"""
		Performs a single update of the Q-Network using the provided optimizer and buffer
		"""
		assert torch.numel(batch.continuous_actions) + torch.numel(batch.discrete_actions) > 0
		continuous_loss = self._compute_continuous_loss(batch, predictions.continuous, targets.continuous)
		discrete_loss = self._compute_discrete_loss(batch, predictions.discrete, targets.discrete)
		loss = continuous_loss + discrete_loss
		# Perform the backpropagation
		self.policy_optimizer.zero_grad()
		loss.backward()
		self.policy_optimizer.step()
		return loss.detach().cpu().numpy().item()
	
	def _compute_policy_ratio(self, batch: BatchExperience) -> torch.Tensor:
		policy_predictions = self.policy.get_logits(batch.obs)
		last_policy_predictions = self._last_policy.get_logits(batch.obs)
		return policy_predictions / (last_policy_predictions + 1e-8)

	def _compute_policy_loss(self, batch: BatchExperience) -> torch.Tensor:
		policy_ratio = self._compute_policy_ratio(batch)
		policy_loss = -torch.mean(
			torch.minimum(
				policy_ratio * batch.advantages,
				torch.clamp(
					policy_ratio,
					1 - self.kwargs["clip_ratio"],
					1 + self.kwargs["clip_ratio"]
				) * batch.advantages
			)
		)
		return policy_loss

	def update_policy_weights(self, batch: BatchExperience) -> float:
		"""
		Performs a single update of the policy network using the provided optimizer and buffer
		"""
		policy_loss = self._compute_policy_loss(batch)
		# Perform the backpropagation
		self.policy_optimizer.zero_grad()
		policy_loss.backward()
		self.policy_optimizer.step()
		return policy_loss.detach().cpu().numpy().item()

