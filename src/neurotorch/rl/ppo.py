import warnings
from typing import Optional, Sequence, Union, Dict, Callable, List, Any

import numpy as np
import scipy
import torch

from .agent import Agent
from .buffers import BatchExperience, Experience
from .utils import discounted_cumulative_sums
from ..transforms.base import to_numpy, to_tensor
from ..learning_algorithms.learning_algorithm import LearningAlgorithm
from ..utils import maybe_apply_softmax


class PPO(LearningAlgorithm):
	r"""
	Apply the Proximal Policy Optimization algorithm to the given model. The algorithm is described in the paper
	`Proximal Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`.
	"""
	CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: str = "optimizer_state_dict"
	
	def __init__(
			self,
			agent: Optional[Agent] = None,
			# params: Optional[Sequence[torch.nn.Parameter]] = None,
			optimizer: Optional[torch.optim.Optimizer] = None,
			**kwargs
	):
		kwargs.setdefault("save_state", True)
		kwargs.setdefault("load_state", True)
		super().__init__(params=None, **kwargs)
		self._agent = agent
		self.last_agent = None
		self.optimizer = optimizer
		self.critic_optimizer = None
		self.policy_params = None
		self.critic_params = None
		self.continuous_criterion = torch.nn.MSELoss()
		self.discrete_criterion = torch.nn.CrossEntropyLoss()
		self.clip_ratio = kwargs.get("clip_ratio", 0.2)
		self.tau = kwargs.get("tau", None)
		self.gamma = kwargs.get("gamma", 0.99)
		self.gae_lambda = kwargs.get("gae_lambda", 0.97)
		self.critic_weight = kwargs.get("critic_weight", 0.5)
		self.kwargs.setdefault("default_lr", 3e-4)
	
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
		self.policy_params = list(self.policy.parameters())
		self.critic_params = list(set(self.critic.parameters()) - set(self.policy_params))
		self.params = list(self.policy_params + self.critic_params)
		param_groups = [
			{"params": self.policy_params, "lr": self.kwargs.get("default_policy_lr", 3e-4)},
			{"params": self.critic_params, "lr": self.kwargs.get("default_critic_lr", 1e-3)}
		]
		if self.optimizer is None:
			self.optimizer = torch.optim.Adam(param_groups)
		# self.critic_optimizer = torch.optim.Adam(self.critic_params, lr=self.kwargs.get("default_critic_lr", 1e-3))

		self.last_agent = trainer.copy_agent()
		if self.tau is None:
			self.tau = 1 / trainer.state.n_epochs
		assert self.tau >= 0, "The parameter `tau` must be greater or equal to 0."
	
	def _compute_policy_ratio(self, batch: BatchExperience) -> torch.Tensor:
		# policy_predictions = self.agent.get_actions(to_tensor(batch.obs), re_format="log_probs", as_numpy=False)
		# with torch.no_grad():
		# 	last_policy_predictions_one_hot, last_policy_predictions_log_smax = self.last_agent.get_actions(
		# 		to_tensor(batch.obs), re_format="one_hot,log_smax", as_numpy=False
		# 	)
		obs_as_tensor = to_tensor(batch.obs)
		policy_preds = self.agent.get_actions(obs_as_tensor, re_format="raw", as_numpy=False)
		with torch.no_grad():
			last_policy_preds = self.last_agent.get_actions(obs_as_tensor, re_format="raw", as_numpy=False)
		
		if isinstance(policy_preds, dict):
			policy_ratio = {}
			for k in policy_preds:
				if k in self.agent.discrete_actions:
					# policy_value = torch.sum(last_policy_predictions_one_hot[k] * policy_predictions[k], dim=-1)
					# policy_ratio[k] = torch.exp(policy_value - last_policy_predictions_log_smax[k])
					policy_dist = torch.distributions.Categorical(
						probs=maybe_apply_softmax(policy_preds[k], dim=-1)
					)
					last_policy_dist = torch.distributions.Categorical(
						probs=maybe_apply_softmax(last_policy_preds[k], dim=-1)
					)
					last_policy_actions = last_policy_dist.sample()
					policy_ratio[k] = torch.exp(
						policy_dist.log_prob(last_policy_actions) - last_policy_dist.log_prob(last_policy_actions)
					)
					policy_ratio[k] = torch.exp(policy_dist.log_prob(last_policy_actions)) / torch.exp(last_policy_dist.log_prob(last_policy_actions))
				else:
					# policy_ratio[k] = policy_predictions[k] / (last_policy_predictions_log_smax[k] + 1e-8)
					policy_ratio[k] = policy_preds[k] / (last_policy_preds[k] + 1e-8)
		elif self.agent.discrete_actions:
			# policy_value = torch.sum(last_policy_predictions_one_hot * policy_predictions, dim=-1)
			# policy_ratio = torch.exp(policy_value - last_policy_predictions_log_smax)
			policy_dist = torch.distributions.Categorical(probs=maybe_apply_softmax(policy_preds, dim=-1))
			last_policy_dist = torch.distributions.Categorical(probs=maybe_apply_softmax(last_policy_preds, dim=-1))
			last_policy_actions = last_policy_dist.sample()
			# policy_ratio = torch.exp(
			# 	policy_dist.log_prob(last_policy_actions) - last_policy_dist.log_prob(last_policy_actions)
			# )
			policy_ratio = torch.exp(policy_dist.log_prob(last_policy_actions)) / torch.exp(
				last_policy_dist.log_prob(last_policy_actions)
				)
		else:
			# policy_ratio = policy_predictions / (last_policy_predictions_log_smax + 1e-8)
			policy_ratio = policy_preds / (last_policy_preds + 1e-8)
		return policy_ratio

	def _compute_policy_loss(self, batch: BatchExperience) -> torch.Tensor:
		policy_ratio = self._compute_policy_ratio(batch)
		advantages = self.get_advantages_from_batch(batch)
		if not isinstance(policy_ratio, dict):
			policy_ratio = {"default": policy_ratio}
		policy_loss = to_tensor(0.0).to(self.policy.device)
		for key, ratio in policy_ratio.items():
			view_shape = [policy_ratio[key].shape[0], ] + (policy_ratio[key].ndim - 1) * [1]
			ratio_adv = policy_ratio[key] * advantages.view(*view_shape).to(self.policy.device)
			ratio_clamped = torch.clamp(policy_ratio[key], 1 - self.clip_ratio, 1 + self.clip_ratio)
			ratio_adv_clamped = ratio_clamped * advantages.view(*view_shape).to(self.policy.device)
			policy_loss += -torch.mean(torch.min(ratio_adv, ratio_adv_clamped))
		return policy_loss
	
	def _compute_critic_loss(self, batch: BatchExperience) -> torch.Tensor:
		critic_predictions = self.critic(to_tensor(batch.obs))
		if isinstance(critic_predictions, dict):
			assert len(critic_predictions) == 1, "Only one critic output is supported."
			critic_values = critic_predictions[list(critic_predictions.keys())[0]].view(-1)
		else:
			critic_values = critic_predictions.view(-1)
		values_targets = self.get_returns_from_batch(batch).view(-1)
		critic_loss = torch.nn.functional.mse_loss(critic_values, values_targets)
		return critic_loss

	def update_params(self, batch: BatchExperience) -> float:
		"""
		Performs a single update of the policy network using the provided optimizer and buffer
		"""
		policy_loss = self._compute_policy_loss(batch)
		critic_loss = self._compute_critic_loss(batch)
		loss = policy_loss + self.critic_weight * critic_loss.to(self.policy.device)
		
		# self.critic_optimizer.zero_grad()
		# critic_loss.backward()
		# self.critic_optimizer.step()
		self.optimizer.zero_grad()
		loss.backward()
		# policy_loss.backward()
		self.optimizer.step()
		
		# loss = policy_loss + self.critic_weight * critic_loss.to(self.policy.device)
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
		terminals = np.asarray([ex.terminal for ex in trajectory])
		values = to_numpy(values).reshape(-1)
		values = np.append(values, (1 - int(terminals[-1])) * values[-1])  # from: https://keras.io/examples/rl/ppo_cartpole/
		rewards = np.array([ex.reward for ex in trajectory.experiences])
		rewards = np.append(rewards, (1 - int(terminals[-1])) * values[-1])  # from: https://keras.io/examples/rl/ppo_cartpole/
		deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
		# deltas = np.append(deltas, rewards[-1] - values[-1])
		advantages = discounted_cumulative_sums(deltas, self.gamma * self.gae_lambda)
		adv_mean, adv_std = np.mean(advantages), np.std(advantages)
		advantages = (advantages - adv_mean) / (adv_std + 1e-8)
		return advantages
	
	def _compute_values(self, trajectory):
		obs_as_tensor = self._batch_obs(trajectory.experiences)
		values = self.agent.get_values(obs_as_tensor, as_numpy=True, re_as_dict=False).reshape(-1)
		return values
	
	def _compute_returns(self, trajectory, values):
		terminals = np.asarray([ex.terminal for ex in trajectory])
		values = to_numpy(values).reshape(-1)
		values = np.append(values, (1 - int(terminals[-1])) * values[-1])  # from: https://keras.io/examples/rl/ppo_cartpole/
		rewards = np.array([ex.reward for ex in trajectory.experiences])
		rewards = np.append(rewards, (1 - int(terminals[-1])) * values[-1])  # from: https://keras.io/examples/rl/ppo_cartpole/
		returns = discounted_cumulative_sums(rewards, self.gamma)[:-1]
		returns_mean, returns_std = returns.mean(), returns.std()
		returns = (returns - returns_mean) / (returns_std + 1e-8)
		return returns
	
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
	
	def get_returns_from_batch(self, batch: BatchExperience) -> torch.Tensor:
		"""
		Computes the returns for the provided batch
		"""
		assert all("return" in x for x in batch.others), "All experiences in the batch must have a return."
		returns = to_tensor([x["return"] for x in batch.others]).to(self.policy.device)
		return returns
	
	def on_optimization_begin(self, trainer, **kwargs):
		super().on_optimization_begin(trainer, **kwargs)
		batch = trainer.current_training_state.x_batch
		batch_loss = self.update_params(batch)
		trainer.update_state_(batch_loss=batch_loss)
	
	def on_optimization_end(self, trainer, **kwargs):
		super().on_optimization_end(trainer, **kwargs)
		if not np.isclose(self.tau, 0.0):
			self.last_policy.soft_update(self.policy, tau=self.tau)

	def on_iteration_begin(self, trainer, **kwargs):
		super().on_iteration_begin(trainer, **kwargs)
		self.last_policy = trainer.copy_policy()
	
	def on_trajectory_end(self, trainer, trajectory, **kwargs) -> List[Dict[str, Any]]:
		super().on_trajectory_end(trainer, trajectory, **kwargs)
		if len(trajectory.experiences) == 0:
			return []
		values = self._compute_values(trajectory)
		advantages = self._compute_advantages(trajectory, values)
		returns = self._compute_returns(trajectory, values)
		trajectory_metrics = [
			{"advantage": advantage, "value": value, "return": returns_item}
			for advantage, value, returns_item in zip(advantages, values, returns)
		]
		trajectory.update_others(trajectory_metrics)
		# for i, exp in enumerate(trajectory.experiences):
		# 	exp.others.update(trajectory_metrics[i])
		# batch_loss = self.update_params(BatchExperience(trajectory.experiences, self.policy.device))
		# trainer.update_state_(batch_loss=batch_loss)
		return trajectory_metrics
	
