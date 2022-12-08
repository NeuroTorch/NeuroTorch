import warnings
from typing import Optional, Sequence, Union, Dict, Callable, List, Any

import numpy as np
import torch

from .agent import Agent
from .buffers import BatchExperience, Experience
from ..transforms.base import to_numpy, to_tensor
from ..learning_algorithms.learning_algorithm import LearningAlgorithm


class PPO(LearningAlgorithm):
	r"""
	Apply the Proximal Policy Optimization algorithm to the given model. The algorithm is described in the paper
	`Proximal Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`.
	"""
	CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: str = "optimizer_state_dict"
	
	def __init__(
			self,
			agent: Optional[Agent] = None,
			params: Optional[Sequence[torch.nn.Parameter]] = None,
			optimizer: Optional[torch.optim.Optimizer] = None,
			**kwargs
	):
		kwargs.setdefault("save_state", True)
		kwargs.setdefault("load_state", True)
		super().__init__(params=params, **kwargs)
		self._agent = agent
		self.last_agent = None
		self.optimizer = optimizer
		self.continuous_criterion = torch.nn.MSELoss()
		self.discrete_criterion = torch.nn.CrossEntropyLoss()
		self.clip_ratio = kwargs.get("clip_ratio", 0.2)
		self.tau = kwargs.get("tau", None)
		self.gamma = kwargs.get("gamma", 0.99)
		self.gae_lambda = kwargs.get("gae_lambda", 0.95)
		self.critic_weight = kwargs.get("critic_weight", 0.5)
		self.kwargs.setdefault("default_lr", 3e-4)
		# TODO: add critic network
	
	@property
	def policy(self):
		if self.agent is None:
			return None
		return self.agent.policy
	
	@property
	def critic(self):
		if self.agent is None:
			return None
		return self.agent.critic
	
	@property
	def last_policy(self):
		if self.last_agent is None:
			return None
		return self.last_agent.policy
	
	@last_policy.setter
	def last_policy(self, policy):
		self.last_agent.policy = policy
	
	@property
	def agent(self):
		if self._agent is not None:
			return self._agent
		if self.trainer is None:
			return None
		return self.trainer.agent
	
	def load_checkpoint_state(self, trainer, checkpoint: dict, **kwargs):
		if self.save_state:
			state = checkpoint.get(self.name, {})
			opt_state_dict = state.get(self.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY, None)
			if opt_state_dict is not None:
				self.optimizer.load_state_dict(opt_state_dict)
	
	def get_checkpoint_state(self, trainer, **kwargs) -> object:
		if self.save_state:
			if self.optimizer is not None:
				return {
					self.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict()
				}
		return None
	
	def start(self, trainer, **kwargs):
		super().start(trainer, **kwargs)
		if self.params and self.optimizer is None:
			self.optimizer = torch.optim.Adam(self.params, lr=self.kwargs["default_lr"])
		elif not self.params and self.optimizer is not None:
			self.params.extend([
				param
				for i in range(len(self.optimizer.param_groups))
				for param in self.optimizer.param_groups[i]["params"]
			])
		else:
			self.params = list(self.policy.parameters()) + list(self.critic.parameters())
			self.optimizer = torch.optim.Adam(self.params, lr=self.kwargs["default_lr"])
		self.last_agent = trainer.copy_agent()
		if self.tau is None:
			self.tau = 1 / trainer.state.n_epochs
		assert self.tau >= 0, "The parameter `tau` must be greater or equal to 0."
	
	def _compute_policy_ratio(self, batch: BatchExperience) -> torch.Tensor:
		policy_predictions = self.agent.get_actions(to_tensor(batch.obs), re_format="probs", as_numpy=False)
		# last_policy_predictions = self.last_agent.get_actions(to_tensor(batch.obs), re_format="probs", as_numpy=False)
		last_policy_predictions = batch.actions
		if isinstance(policy_predictions, dict):
			policy_ratio = {
				key: policy_predictions[key] / (last_policy_predictions[key] + 1e-8)
				for key in policy_predictions
			}
		else:
			policy_ratio = policy_predictions / (last_policy_predictions + 1e-8)
		return policy_ratio

	def _compute_policy_loss(self, batch: BatchExperience) -> torch.Tensor:
		policy_ratio = self._compute_policy_ratio(batch)
		advantages = self.get_advantages_from_batch(batch)
		if not isinstance(policy_ratio, dict):
			policy_ratio = {"default": policy_ratio}
		policy_loss = to_tensor(0.0).to(self.policy.device)
		for key, ratio in policy_ratio.items():
			view_shape = [policy_ratio[key].shape[0], ] + (policy_ratio[key].ndim - 1) * [1]
			policy_loss += -torch.mean(
				torch.minimum(
					policy_ratio[key] * advantages.view(*view_shape).to(self.policy.device),
					torch.clamp(
						policy_ratio[key],
						1 - self.clip_ratio,
						1 + self.clip_ratio
					) * advantages.view(*view_shape).to(self.policy.device)
				)
			)
		return policy_loss
	
	def _compute_critic_loss(self, batch: BatchExperience) -> torch.Tensor:
		advantages = self.get_advantages_from_batch(batch).to(self.critic.device)
		values = self.get_values_from_batch(batch).to(self.critic.device)
		critic_predictions = self.critic(to_tensor(batch.obs))
		if isinstance(critic_predictions, dict):
			assert len(critic_predictions) == 1, "Only one critic output is supported."
			critic_values = critic_predictions[list(critic_predictions.keys())[0]].view(-1)
		else:
			critic_values = critic_predictions.view(-1)
		values_targets = advantages.view(-1) + values.view(-1)
		critic_loss = torch.functional.F.mse_loss(critic_values, values_targets)
		return critic_loss

	def update_policy_weights(self, batch: BatchExperience) -> float:
		"""
		Performs a single update of the policy network using the provided optimizer and buffer
		"""
		policy_loss = self._compute_policy_loss(batch)
		critic_loss = self._compute_critic_loss(batch)
		loss = policy_loss + self.critic_weight * critic_loss.to(self.policy.device)
		# Perform the backpropagation
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return to_numpy(loss).item()
	
	def _batch_obs(self, batch: List[Experience]):
		as_dict = isinstance(batch[0].obs, dict)
		if as_dict:
			obs_batched = batch[0].obs
			for key in obs_batched:
				obs_batched[key] = torch.stack([to_tensor(ex.obs[key]) for ex in batch]).to(self.policy.device)
		else:
			obs_batched = torch.stack([to_tensor(ex.obs) for ex in batch]).to(self.policy.device)
		return obs_batched
	
	def _compute_advantages(self, trajectory, values):
		advantages = np.zeros(len(trajectory))
		advantages[-1] = trajectory.experiences[-1].reward - values[-1]
		T = len(trajectory.experiences) - 1
		for i in reversed(range(T)):
			delta = trajectory.experiences[i].reward + self.gamma * values[i + 1] - values[i]
			advantages[i] = delta + self.gamma * self.gae_lambda * advantages[i + 1]
		
		adv_mean, adv_std = np.mean(advantages), np.std(advantages)
		advantages = (advantages - adv_mean) / (adv_std + 1e-8)
		return advantages
	
	def _compute_values(self, trajectory):
		obs_as_tensor = self._batch_obs(trajectory.experiences)
		values = self.agent.get_values(obs_as_tensor, as_numpy=True, re_as_dict=False).reshape(-1)
		return values
	
	def get_advantages_from_batch(self, batch: BatchExperience) -> torch.Tensor:
		"""
		Computes the advantages for the provided batch
		"""
		assert all("advantage" in x for x in batch.others), "All experiences in the batch must have an advantage."
		advantages = to_tensor([x["advantage"] for x in batch.others]).to(self.policy.device)
		return advantages
	
	def get_values_from_batch(self, batch: BatchExperience) -> torch.Tensor:
		"""
		Computes the values for the provided batch
		"""
		assert all("value" in x for x in batch.others), "All experiences in the batch must have a value."
		values = to_tensor([x["value"] for x in batch.others]).to(self.policy.device)
		return values
	
	def on_optimization_begin(self, trainer, **kwargs):
		super().on_optimization_begin(trainer, **kwargs)
		batch = trainer.current_training_state.x_batch
		batch_loss = self.update_policy_weights(batch)
		trainer.update_state_(batch_loss=batch_loss)
	
	def on_optimization_end(self, trainer, **kwargs):
		super().on_optimization_end(trainer, **kwargs)
		if not np.isclose(self.tau, 0.0):
			self.last_policy.soft_update(self.policy, tau=self.tau)

	def on_iteration_begin(self, trainer, **kwargs):
		super().on_iteration_begin(trainer, **kwargs)
		self.last_policy = trainer.copy_policy()
	
	def on_trajectory_end(self, trainer, trajectory, **kwargs) -> List[Dict[str, Any]]:
		# TODO: problem: if the trajectory is not finished before the optimization, the advantages are never computed
		super().on_trajectory_end(trainer, trajectory, **kwargs)
		values = self._compute_values(trajectory)
		advantages = self._compute_advantages(trajectory, values)
		return [{"advantage": advantage, "value": value} for advantage, value in zip(advantages, values)]
		
