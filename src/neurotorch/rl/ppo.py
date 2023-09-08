import warnings
from typing import Optional, Sequence, Union, Dict, Callable, List, Any

import numpy as np
import scipy
import torch

from .agent import Agent
from .buffers import BatchExperience, Experience
from .utils import discounted_cumulative_sums, continuous_actions_distribution
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
            optimizer: Optional[torch.optim.Optimizer] = None,
            **kwargs
    ):
        """
        Constructor of the PPO algorithm.

        :param agent: The agent to train.
        :type agent: Agent
        :param optimizer: The optimizer to use.
        :type optimizer: torch.optim.Optimizer
        :param kwargs: Additional keyword arguments.

        :keyword float clip_ratio: The clipping ratio for the policy loss.
        :keyword float tau: The smoothing factor for the policy update.
        :keyword float gamma: The discount factor.
        :keyword float gae_lambda: The lambda parameter for the generalized advantage estimation (GAE).
        :keyword float critic_weight: The weight of the critic loss.
        :keyword float entropy_weight: The weight of the entropy loss.
        :keyword torch.nn.Module critic_criterion: The loss function to use for the critic.
        :keyword bool advantages=returns-values: This keyword is introduced to fix a bug when using the GAE. If set to
            True, the advantages are computed as the returns minus the values. If set to False, the advantages are
            compute as in the PPO paper. The default value is False and it is recommended to try to set it to True
            if the agent doesn't seem to learn.
        :keyword float max_grad_norm: The maximum L2 norm of the gradient. Default is 0.5.
        """
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
        self.critic_clip = kwargs.get("critic_clip", 0.2)
        self.tau = kwargs.get("tau", 0.0)
        self.gamma = kwargs.get("gamma", 0.99)
        self.gae_lambda = kwargs.get("gae_lambda", 0.99)
        self.critic_weight = kwargs.get("critic_weight", 0.5)
        self.entropy_weight = kwargs.get("entropy_weight", 0.01)
        self.critic_criterion = kwargs.get("critic_criterion", torch.nn.MSELoss())
        self.adv_as_returns_values = kwargs.get("advantages=returns-values", False)
        self.max_grad_norm = kwargs.get("max_grad_norm", 0.5)

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
        self.policy_params = list(filter(lambda p: p.requires_grad, self.policy.parameters()))
        self.critic_params = list(filter(lambda p: p.requires_grad, self.critic.parameters()))
        self.params = list(self.policy_params + self.critic_params)
        param_groups = [
            {"params": self.policy_params, "lr": self.kwargs.get("default_policy_lr", 2e-4)},
            {"params": self.critic_params, "lr": self.kwargs.get("default_critic_lr", 1e-3)},
        ]
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(param_groups)

        self.last_agent = trainer.copy_agent()
        if self.tau is None:
            self.tau = 1 / trainer.state.n_epochs
        assert self.tau >= 0, "The parameter `tau` must be greater or equal to 0."

    def _compute_policy_distributions(self, batch: BatchExperience, **kwargs):
        """
        Computes the policy distributions for the given batch.

        :param batch: The batch to compute the policy distributions for.
        :param kwargs: Other keyword arguments.

        :return: The policy distributions and the last policy distributions.
        """
        obs_as_tensor = to_tensor(batch.obs)
        policy_preds = kwargs.get(
            "policy_preds", self.agent.get_actions(obs_as_tensor, re_format="raw", as_numpy=False)
        )
        with torch.no_grad():
            last_policy_preds = self.last_agent.get_actions(obs_as_tensor, re_format="raw", as_numpy=False)

        if isinstance(policy_preds, dict):
            policy_dist, last_policy_dist = {}, {}
            for k in policy_preds:
                if k in self.agent.discrete_actions:
                    policy_dist[k] = torch.distributions.Categorical(
                        probs=maybe_apply_softmax(policy_preds[k], dim=-1)
                    )
                    last_policy_dist[k] = torch.distributions.Categorical(
                        probs=maybe_apply_softmax(last_policy_preds[k], dim=-1)
                    )
                else:
                    # TODO: must get the right covariance for each continuous action, see :class:`Agent`.
                    covariance = self.agent.get_continuous_action_covariances()[self.agent.continuous_actions[0]]
                    policy_dist[k] = continuous_actions_distribution(policy_preds[k], covariance=covariance)
                    last_policy_dist[k] = continuous_actions_distribution(last_policy_preds[k], covariance=covariance)
        elif self.agent.discrete_actions:
            policy_dist = torch.distributions.Categorical(probs=maybe_apply_softmax(policy_preds, dim=-1))
            last_policy_preds_smax = maybe_apply_softmax(last_policy_preds, dim=-1)
            last_policy_dist = torch.distributions.Categorical(probs=last_policy_preds_smax)
        else:
            covariance = self.agent.get_continuous_action_covariances()[self.agent.continuous_actions[0]]
            policy_dist = continuous_actions_distribution(policy_preds, covariance=covariance)
            last_policy_dist = continuous_actions_distribution(last_policy_preds, covariance=covariance)
        return policy_dist, last_policy_dist

    def _compute_policy_ratio(self, batch: BatchExperience, **kwargs) -> torch.Tensor:
        obs_as_tensor = to_tensor(batch.obs)
        actions = self.get_actions_from_batch(batch)
        policy_preds = kwargs.get("policy_preds", self.agent.get_actions(obs_as_tensor, re_format="raw", as_numpy=False))
        with torch.no_grad():
            last_policy_preds = self.last_agent.get_actions(obs_as_tensor, re_format="raw", as_numpy=False)

        if isinstance(policy_preds, dict):
            policy_ratio = {}
            for k in policy_preds:
                key_actions = actions[k] if isinstance(actions, dict) else actions
                if k in self.agent.discrete_actions:
                    policy_dist = torch.distributions.Categorical(
                        probs=maybe_apply_softmax(policy_preds[k], dim=-1)
                    )
                    last_policy_dist = torch.distributions.Categorical(
                        probs=maybe_apply_softmax(last_policy_preds[k], dim=-1)
                    )
                else:
                    policy_dist = continuous_actions_distribution(policy_preds[k])
                    last_policy_dist = continuous_actions_distribution(last_policy_preds[k])
                # policy_ratio[k] = policy_preds[k] / (last_policy_preds[k] + 1e-8)
                policy_ratio[k] = torch.exp(
                    policy_dist.log_prob(key_actions) - last_policy_dist.log_prob(key_actions)
                )
        elif self.agent.discrete_actions:
            key_actions = actions[list(actions.keys())[0]] if isinstance(actions, dict) else actions
            policy_dist = torch.distributions.Categorical(probs=maybe_apply_softmax(policy_preds, dim=-1))
            last_policy_preds_smax = maybe_apply_softmax(last_policy_preds, dim=-1)
            last_policy_dist = torch.distributions.Categorical(probs=last_policy_preds_smax)
            policy_ratio = torch.exp(
                policy_dist.log_prob(key_actions) - last_policy_dist.log_prob(key_actions)
            )
        else:
            policy_dist = continuous_actions_distribution(policy_preds)
            last_policy_dist = continuous_actions_distribution(last_policy_preds)
            key_actions = actions[list(actions.keys())[0]] if isinstance(actions, dict) else actions
            policy_ratio = torch.exp(
                policy_dist.log_prob(key_actions) - last_policy_dist.log_prob(key_actions)
            )
        # policy_ratio = policy_preds / (last_policy_preds + 1e-8)
        return policy_ratio

    def _compute_policy_ratio_from_distributions(
            self, batch: BatchExperience, policy_dist, last_policy_dist, **kwargs
    ):
        actions = self.get_actions_from_batch(batch)
        if isinstance(policy_dist, dict):
            policy_ratio = {}
            for k in policy_dist:
                key_actions = actions[k] if isinstance(actions, dict) else actions
                policy_ratio[k] = torch.exp(
                    policy_dist[k].log_prob(key_actions) - last_policy_dist[k].log_prob(key_actions)
                )
        elif self.agent.discrete_actions:
            key_actions = actions[list(actions.keys())[0]] if isinstance(actions, dict) else actions
            policy_ratio = torch.exp(
                policy_dist.log_prob(key_actions) - last_policy_dist.log_prob(key_actions)
            )
        else:
            key_actions = actions[list(actions.keys())[0]] if isinstance(actions, dict) else actions
            policy_ratio = torch.exp(
                policy_dist.log_prob(key_actions) - last_policy_dist.log_prob(key_actions)
            )
        return policy_ratio

    def _compute_policy_loss(self, batch: BatchExperience, **kwargs) -> torch.Tensor:
        # policy_ratio = self._compute_policy_ratio(batch, **kwargs)
        policy_ratio = self._compute_policy_ratio_from_distributions(batch, **kwargs)
        advantages = self.get_advantages_from_batch(batch)
        if not isinstance(policy_ratio, dict):
            policy_ratio = {"default": policy_ratio}
        policy_loss = to_tensor(0.0).to(self.policy.device)
        for key, ratio in policy_ratio.items():
            view_shape = [policy_ratio[key].shape[0], ] + (policy_ratio[key].ndim - 1) * [1]
            ratio_adv = policy_ratio[key] * advantages.view(*view_shape).to(self.policy.device)
            ratio_clamped = torch.clamp(policy_ratio[key], min=1.0 - self.clip_ratio, max=1.0 + self.clip_ratio)
            ratio_adv_clamped = ratio_clamped * advantages.view(*view_shape).to(self.policy.device)
            policy_loss += -torch.mean(torch.min(ratio_adv, ratio_adv_clamped))
        return policy_loss

    def _compute_critic_loss(self, batch: BatchExperience, **kwargs) -> torch.Tensor:
        """
        Compute the critic loss.

        The math are taken from https://github.com/Unity-Technologies/ml-agents/blob/6bb711f1b0a29a06b4deb544bb1e93c392acb255/ml-agents/mlagents/trainers/torch_entities/utils.py#L400.

        :param batch:
        :param kwargs:
        :return:
        """
        critic_predictions = self.critic(to_tensor(batch.obs))
        with torch.no_grad():
            last_critic_preds = self.last_agent.critic(to_tensor(batch.obs))
        if isinstance(critic_predictions, dict):
            assert len(critic_predictions) == 1, "Only one critic output is supported."
            critic_values = critic_predictions[list(critic_predictions.keys())[0]].view(-1)
            last_critic_preds = last_critic_preds[list(last_critic_preds.keys())[0]].view(-1)
        else:
            critic_values = critic_predictions.view(-1)
            last_critic_preds = last_critic_preds.view(-1)
        values_targets = self.get_returns_from_batch(batch).view(-1)
        clipped_value_estimate = last_critic_preds + torch.clamp(
            critic_values - last_critic_preds, -1 * self.critic_clip, self.critic_clip
        )
        v_opt_a = self.critic_criterion(critic_values, values_targets)
        v_opt_b = self.critic_criterion(critic_values, clipped_value_estimate)
        critic_loss = torch.mean(torch.max(v_opt_a, v_opt_b))
        return critic_loss

    def _compute_entropy_loss(self, batch: BatchExperience, **kwargs) -> torch.Tensor:
        obs_as_tensor = to_tensor(batch.obs)
        policy_preds = kwargs.get("policy_preds", self.agent.get_actions(obs_as_tensor, re_format="raw", as_numpy=False))
        if isinstance(policy_preds, dict):
            entropy_loss = to_tensor(0.0).to(self.policy.device)
            for k in policy_preds:
                if k in self.agent.discrete_actions:
                    policy_dist = torch.distributions.Categorical(
                        probs=maybe_apply_softmax(policy_preds[k], dim=-1)
                    )
                    entropy_loss += torch.mean(policy_dist.entropy())
                else:
                    # TODO: implement covariance matrix to compute entropy
                    # policy_dist = torch.distributions.MultivariateNormal(policy_preds[k], torch.eye(policy_preds[k].shape[-1]))
                    # entropy_loss += torch.mean(policy_dist.entropy())
                    pass
        elif self.agent.discrete_actions:
            policy_dist = torch.distributions.Categorical(probs=maybe_apply_softmax(policy_preds, dim=-1))
            entropy_loss = torch.mean(policy_dist.entropy())
        else:
            # TODO: implement covariance matrix to compute entropy
            # policy_dist = torch.distributions.MultivariateNormal(policy_preds, torch.eye(policy_preds.shape[-1]))
            # entropy_loss = torch.mean(policy_dist.entropy())
            entropy_loss = to_tensor(0.0).to(self.policy.device)
        return entropy_loss

    def _compute_entropy_loss_from_distributions(
            self, batch: BatchExperience, **kwargs
    ) -> torch.Tensor:
        obs_as_tensor = to_tensor(batch.obs)
        policy_preds = kwargs.get(
            "policy_preds", self.agent.get_actions(obs_as_tensor, re_format="raw", as_numpy=False)
        )
        kwargs["policy_preds"] = policy_preds
        policy_dist = kwargs.get("policy_dist", self._compute_policy_distributions(batch, **kwargs)[0])
        if isinstance(policy_dist, dict):
            entropy_loss = to_tensor(0.0).to(self.policy.device)
            for k in policy_dist:
                entropy_loss += torch.mean(policy_dist[k].entropy())
        else:
            entropy_loss = torch.mean(policy_dist.entropy())
        return entropy_loss

    def update_params(self, batch: BatchExperience) -> float:
        """
        Performs a single update of the policy network using the provided optimizer and buffer
        """
        policy_preds = self.agent.get_actions(to_tensor(batch.obs), re_format="raw", as_numpy=False)
        policy_dist, last_policy_dist = self._compute_policy_distributions(batch, policy_preds=policy_preds)
        policy_loss = self._compute_policy_loss(
            batch, policy_preds=policy_preds, policy_dist=policy_dist, last_policy_dist=last_policy_dist
        )
        critic_loss = self._compute_critic_loss(batch)
        entropy_loss = self._compute_entropy_loss_from_distributions(
            batch, policy_preds=policy_preds, policy_dist=policy_dist, last_policy_dist=last_policy_dist
        )
        weighted_critic_loss = self.critic_weight * critic_loss.to(self.policy.device)
        weighted_entropy_loss = self.entropy_weight * entropy_loss.to(self.policy.device)
        loss = policy_loss + weighted_critic_loss - weighted_entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()
        self.agent.decay_continuous_action_variances()

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
        values = to_numpy(values).reshape(-1)
        if self.adv_as_returns_values:
            advantages = self._compute_returns(trajectory, values) - values
        else:
            terminals = np.asarray([ex.terminal for ex in trajectory])
            rewards = np.array([ex.reward for ex in trajectory.experiences])
            rewards = np.append(rewards, (1 - int(terminals[-1])) * values[-1])  # from: https://keras.io/examples/rl/ppo_cartpole/
            values = np.append(values, (1 - int(terminals[-1])) * values[-1])  # from: https://keras.io/examples/rl/ppo_cartpole/
            deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
            advantages = discounted_cumulative_sums(deltas, self.gamma * self.gae_lambda)

        adv_mean, adv_std = np.mean(advantages), np.std(advantages)
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        return advantages

    def _compute_values(self, trajectory):
        obs_as_tensor = BatchExperience(trajectory).obs
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

    def get_actions_from_batch(self, batch: BatchExperience) -> torch.Tensor:
        """
        Get the actions for the provided batch
        """
        actions = batch.actions
        if isinstance(actions, dict):
            for key in actions:
                if key in self.agent.discrete_actions:
                    actions[key] = actions[key].to(self.policy.device)
                    if actions[key].dim() > 1 and actions[key].shape[-1] > 1:
                        actions[key] = torch.argmax(actions[key], dim=-1)
                    actions[key] = actions[key].long()
        elif self.agent.discrete_actions:
            if actions.dim() > 1 and actions.shape[-1] > 1:
                actions = torch.argmax(actions, dim=-1)
            actions = actions.long()
        return actions

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
        self.last_policy.hard_update(self.policy)

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

    def on_pbar_update(self, trainer, **kwargs) -> dict:
        """
        Called when the progress bar is updated.

        :param trainer: The trainer.
        :type trainer: Trainer
        :param kwargs: Additional arguments.

        :return: None
        """
        repr_action_var = {
            k: [float(f"{v:.3f}") for v in self.agent.continuous_action_variances[k].data.tolist()]
            for k in self.agent.continuous_action_variances
        }
        return {"actions_var": repr_action_var}
	
