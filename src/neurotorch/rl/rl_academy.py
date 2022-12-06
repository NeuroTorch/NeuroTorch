import os
import shutil
import time
import warnings
from collections import defaultdict, deque, OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .agent import Agent
from .buffers import ReplayBuffer, Trajectory, Experience, BatchExperience
from .ppo import PPO
from .utils import env_batch_step
from .. import Trainer, LoadCheckpointMode, to_numpy, TrainingHistory
from ..callbacks.base_callback import BaseCallback, CallbacksList
from ..learning_algorithms.learning_algorithm import LearningAlgorithm
from ..modules import BaseModel
from ..utils import linear_decay


class AgentsHistoryMaps:
	r"""
	Class to store the mapping between agents and their history maps

	Attributes:
		trajectories (Dict[int, Trajectory]): Mapping between agent ids and their trajectories
		cumulative_rewards (Dict[int, float]): Mapping between agent ids and their cumulative rewards
	"""

	def __init__(self, buffer: Optional[ReplayBuffer] = None):
		self.buffer = buffer if buffer is not None else ReplayBuffer()
		self.trajectories: Dict[int, Trajectory] = defaultdict(Trajectory)
		self.cumulative_rewards: Dict[int, float] = defaultdict(lambda: 0.0)
		self._terminal_counter = 0

	@property
	def terminals_count(self) -> int:
		"""
		:return: The number of terminal steps
		"""
		return self._terminal_counter
	
	def update_trajectories_(
			self,
			*,
			observations,
			actions,
			next_observations,
			rewards,
			dones,
			truncated=None,
			infos=None
	):
		actions = deepcopy(to_numpy(actions))
		observations, next_observations = deepcopy(to_numpy(observations)), deepcopy(to_numpy(next_observations))
		rewards, dones = deepcopy(to_numpy(rewards)), deepcopy(to_numpy(dones))
		for i in range(len(dones)):
			if self.trajectories[i].terminated:
				continue
			if dones[i]:
				self.trajectories[i].append_and_terminate(Experience(
					obs=observations[i],
					reward=rewards[i],
					terminal=dones[i],
					action=actions[i],
					next_obs=next_observations[i],
				))
				self.cumulative_rewards[i] = self.trajectories[i].cumulative_reward
				self.buffer.extend(self.trajectories.pop(i))
				self._terminal_counter += 1
			else:
				self.trajectories[i].append(Experience(
					obs=observations[i],
					reward=rewards[i],
					terminal=dones[i],
					action=actions[i],
					next_obs=next_observations[i],
				))
				
	def terminate_all(self):
		for i in range(len(self.trajectories)):
			if not self.trajectories[i].terminated:
				self.trajectories[i].terminate()
			self.cumulative_rewards[i] = self.trajectories[i].cumulative_reward
			self.buffer.extend(self.trajectories.pop(i))
			self._terminal_counter += 1


class RLAcademy(Trainer):
	REWARD_METRIC_KEY = "rewards"
	
	def __init__(
			self,
			agent: Agent,
			*,
			predict_method: str = "__call__",
			learning_algorithm: Optional[LearningAlgorithm] = None,
			callbacks: Optional[Union[List[BaseCallback], CallbacksList, BaseCallback]] = None,
			verbose: bool = True,
			**kwargs
	):
		self.agent = agent
		kwargs = self._set_default_academy_kwargs(**kwargs)
		super().__init__(
			model=agent.policy,
			predict_method=predict_method,
			learning_algorithm=learning_algorithm,
			callbacks=callbacks,
			verbose=verbose,
			**kwargs
		)
	
	@property
	def env(self):
		if "env" not in self.state.objects:
			raise ValueError("The environment is not set.")
		return self.state.objects["env"]
	
	@property
	def policy(self) -> BaseModel:
		"""
		:return: The policy of the academy.
		"""
		return self.model

	@staticmethod
	def _set_default_academy_kwargs(**kwargs) -> Dict[str, Any]:
		"""
		Set default values for the kwargs of the fit method.
		:param kwargs:
			close_env: Whether to close the environment after the training.
			n_epochs: Number of epochs to train each iteration.
			init_lr: Initial learning rate.
			min_lr: Minimum learning rate.
			weight_decay: Weight decay.
			init_epsilon: Initial epsilon. Epsilon is the probability of choosing a random action.
			epsilon_decay: Epsilon decay.
			min_epsilon: Minimum epsilon.
			gamma: Discount factor.
			tau: Target network update rate.
			n_batches: Number of batches to train each iteration.
			batch_size: Batch size.
			update_freq: Number of steps between each update.
			curriculum_strength: Strength of the teacher learning strategy.
		:return:
		"""
		kwargs.setdefault("close_env", False)
		kwargs.setdefault("init_epsilon", 0.01)
		kwargs.setdefault("epsilon_decay", 0.995)
		kwargs.setdefault("min_epsilon", 0.0)
		kwargs.setdefault("n_batches", 3)
		kwargs.setdefault("tau", 1/kwargs["n_batches"])
		kwargs.setdefault("batch_size", 256)
		kwargs.setdefault("n_new_trajectories", 32)
		kwargs.setdefault("buffer_size", 4096)
		kwargs.setdefault("clip_ratio", 0.2)
		kwargs.setdefault("use_priority_buffer", True)

		assert kwargs["batch_size"] <= kwargs["buffer_size"]
		assert kwargs["batch_size"] > 0
		return kwargs
	
	def _maybe_add_learning_algorithm(self, learning_algorithm: Optional[LearningAlgorithm]) -> None:
		if len(self.learning_algorithms) == 0 and learning_algorithm is None:
			learning_algorithm = PPO(optimizer=self.optimizer, criterion=self.criterion)
		if learning_algorithm is not None:
			self.callbacks.append(learning_algorithm)

	def copy_policy(self, requires_grad: bool = False) -> BaseModel:
		"""
		Copy the policy to a new instance.
		:return: The copied policy.
		"""
		policy_copy = deepcopy(self.policy)
		for param in policy_copy.parameters():
			param.requires_grad = requires_grad
		policy_copy.eval()
		return policy_copy

	def _update_optimizer_(self):
		self.policy_optimizer = torch.optim.Adam(
			self.policy.parameters(),
			lr=self.kwargs["init_lr"],
			weight_decay=self.kwargs["weight_decay"]
		)

	def generate_trajectories(
			self,
			n_trajectories: Optional[int] = None,
			buffer: Optional[ReplayBuffer] = None,
			epsilon: float = 0.0,
			p_bar_position: int = 0,
			verbose: Optional[bool] = None,
			**kwargs
	) -> Tuple[ReplayBuffer, List[float]]:
		if n_trajectories is None:
			n_trajectories = self.kwargs["n_new_trajectories"]
		if buffer is None:
			buffer = ReplayBuffer(self.kwargs["buffer_size"], use_priority=self.kwargs["use_priority_buffer"])
		if verbose is None:
			verbose = self.verbose
		if "env" in kwargs:
			self.update_objects_state_(env=kwargs["env"])
		agents_history_maps = AgentsHistoryMaps(buffer)
		cumulative_rewards: List[float] = []
		p_bar = tqdm(
			total=n_trajectories, disable=not verbose, desc="Generating Trajectories", position=p_bar_position
		)
		observations, info = self.env.reset()
		while agents_history_maps.terminals_count < n_trajectories:  # While not enough data in the buffer
			if np.random.random() < epsilon:
				actions = self.agent.get_random_actions(env=self.env)
			else:
				actions = self.agent.get_actions(observations, env=self.env)
			next_observations, rewards, dones, truncated, infos = env_batch_step(self.env, actions)
			agents_history_maps.update_trajectories_(
				observations=observations,
				actions=actions,
				next_observations=next_observations,
				rewards=rewards,
				dones=dones,
			)
			cumulative_rewards = list(agents_history_maps.cumulative_rewards.values())
			if all(dones):
				agents_history_maps.terminate_all()
				next_observations, info = self.env.reset()
			p_bar.update(min(sum(dones), max(0, n_trajectories - sum(dones))))
			p_bar.set_postfix(cumulative_reward=f"{np.mean(cumulative_rewards) if cumulative_rewards else 0.0:.3f}")
			observations = next_observations
		
		p_bar.close()
		return buffer, cumulative_rewards

	def _init_train_buffer(self) -> ReplayBuffer:
		self.env.reset()
		buffer, _ = self.generate_trajectories(self.kwargs["batch_size"])
		return buffer

	def train(
			self,
			env,
			n_iterations: Optional[int] = None,
			*,
			n_epochs: int = 1,
			load_checkpoint_mode: LoadCheckpointMode = None,
			force_overwrite: bool = False,
			p_bar_position: Optional[int] = None,
			p_bar_leave: Optional[bool] = None,
			**kwargs
	) -> TrainingHistory:
		self._load_checkpoint_mode = load_checkpoint_mode
		self._force_overwrite = force_overwrite
		self.kwargs.update(kwargs)
		self.update_state_(
			n_iterations=n_iterations,
			n_epochs=n_epochs,
			objects={**self.current_training_state.objects, **{"env": env}}
		)
		self.sort_callbacks_()
		self.callbacks.start(self)
		self.load_state()
		if self.current_training_state.iteration is None:
			self.update_state_(iteration=0)
		if len(self.training_history) > 0:
			self.update_itr_metrics_state_(**self.training_history.get_item_at(-1))
		else:
			self.update_state_(itr_metrics={})
		env.reset()
		buffer = self._init_train_buffer()
		self.update_objects_state_(buffer=buffer)
		p_bar = tqdm(
			initial=self.current_training_state.iteration,
			total=self.current_training_state.n_iterations,
			desc=kwargs.get("desc", "Training"),
			disable=not self.verbose,
			position=p_bar_position,
			unit="itr",
			leave=p_bar_leave
		)
		# last_save_rewards = deque(maxlen=save_freq)
		for i in self._iterations_generator(p_bar):
			self.update_state_(iteration=i)
			self.callbacks.on_iteration_begin(self)
			epsilon = linear_decay(
				self.kwargs["init_epsilon"], self.kwargs["min_epsilon"], self.kwargs["epsilon_decay"], i
			)
			env = self.current_training_state.objects["env"]
			buffer = self.current_training_state.objects["buffer"]
			env.reset()
			itr_loss = self._exec_iteration(env, buffer, epsilon=epsilon)
			self.update_itr_metrics_state_(**itr_loss, epsilon=epsilon)
			postfix = {f"{k}": f"{v:.5e}" for k, v in self.state.itr_metrics.items()}
			postfix.update(self.callbacks.on_pbar_update(self))
			self.callbacks.on_iteration_end(self)
			p_bar.set_postfix(postfix)
			if self.current_training_state.stop_training_flag:
				p_bar.set_postfix(OrderedDict(**{"stop_flag": "True"}, **postfix))
				break
			# teacher_loss = self.fit_curriculum_buffer()
			# itr_metrics = self._exec_fit_itr_(epsilon, buffer)
			# best_rewards = max(best_rewards, itr_metrics["Rewards"])
			# last_save_rewards.append(itr_metrics["Rewards"])
			# p_bar_postfix = {
			# 	"loss": f"{itr_metrics['Loss']:.3f}",
			# 	"itr_rewards": f"{itr_metrics['Rewards']:.3f}",
			# 	f"last_{save_freq}_rewards": f"{np.mean(last_save_rewards):.3f}",
			# 	"best_rewards": f"{best_rewards:.3f}",
			# }
		
		self.callbacks.close(self)
		p_bar.close()
		if self.kwargs.get("close_env", True):
			env.close()
		return self.training_history
	
	def _exec_iteration(
			self,
			env: gym.Env,
			buffer: ReplayBuffer,
			**kwargs,
	) -> Dict[str, float]:
		with torch.no_grad():
			torch.cuda.empty_cache()
		losses = {}
		
		self.model.train()
		self.callbacks.on_train_begin(self)
		
		buffer, cumulative_rewards = self.generate_trajectories(
			self.kwargs["n_new_trajectories"], buffer, kwargs.get("epsilon", 0.0),
			p_bar_position=0, verbose=False,
		)
		losses[self.REWARD_METRIC_KEY] = np.mean(cumulative_rewards)
		self.update_state_(batch_is_train=True)
		train_losses = []
		for epoch_idx in range(self.current_training_state.n_epochs):
			self.update_state_(epoch=epoch_idx)
			train_losses.append(self._exec_epoch(buffer))
		train_loss = np.mean(train_losses)
		self.update_state_(train_loss=train_loss)
		self.callbacks.on_train_end(self)
		losses["train_loss"] = train_loss
		
		# if val_dataloader is not None:
		# 	with torch.no_grad():
		# 		self.model.eval()
		# 		self.callbacks.on_validation_begin(self)
		# 		self.update_state_(batch_is_train=False)
		# 		val_loss = self._exec_epoch(val_dataloader)
		# 		self.update_state_(val_loss=val_loss)
		# 		self.callbacks.on_validation_end(self)
		# 		losses["val_loss"] = val_loss
		
		with torch.no_grad():
			torch.cuda.empty_cache()
		return losses
	
	def _exec_epoch(
			self,
			buffer: ReplayBuffer,
	) -> float:
		self.callbacks.on_epoch_begin(self)
		batch_size = min(len(buffer), self.kwargs["batch_size"])
		batches = buffer.get_batch_generator(
			batch_size, self.kwargs["n_batches"], randomize=True, device=self.agent.policy.device
		)
		batch_losses = []
		for i, exp_batch in enumerate(batches):
			self.update_state_(batch=i)
			batch_losses.append(to_numpy(self._exec_batch(exp_batch)))
		mean_loss = np.mean(batch_losses)
		self.callbacks.on_epoch_end(self)
		return mean_loss
	
	def _exec_batch(
			self,
			exp_batch: BatchExperience,
			**kwargs,
	):
		self.update_state_(x_batch=exp_batch)
		self.callbacks.on_batch_begin(self)
		if self.model.training:
			self.callbacks.on_optimization_begin(self, x=exp_batch)
			self.callbacks.on_optimization_end(self)
		else:
			self.callbacks.on_validation_batch_begin(self, x=exp_batch)
			self.callbacks.on_validation_batch_end(self)
		self.callbacks.on_batch_end(self)
		batch_loss = self.current_training_state.batch_loss
		if batch_loss is None:
			batch_loss = 0.0
		elif hasattr(batch_loss, "item") and callable(batch_loss.item):
			batch_loss = batch_loss.item()
		return batch_loss

	def _update_bc_optimizer_(self):
		if self.curriculum.teacher_buffer is None:
			return None
		base_lr = self.kwargs["lr"]
		bc_strength = self.kwargs["bc_strength"]
		if self.curriculum.current_lesson.teacher_strength is not None:
			bc_strength = self.curriculum.current_lesson.teacher_strength
		for g in self.cloning_optimizer.param_groups:
			g['lr'] = base_lr * bc_strength

	def _exec_fit_itr_(
			self,
			epsilon: float,
			buffer: ReplayBuffer,
	) -> Dict[str, float]:
		buffer, cumulative_rewards = self.generate_trajectories(
			self.kwargs["update_freq"], buffer, epsilon, p_bar_position=0, verbose=False,
		)
		cum_rewards = np.mean(cumulative_rewards)
		itr_loss = self.fit_buffer(buffer)
		return dict(Rewards=cum_rewards, Loss=itr_loss)

	def fit_curriculum_buffer(self) -> Optional[float]:
		"""
		Fit the curriculum buffer.
		
		:return: The teacher loss.
		"""
		buffer = self.curriculum.teacher_buffer
		if buffer is None:
			return None
		batch_size = min(len(buffer), self.kwargs["batch_size"])
		batches = buffer.get_batch_generator(
			batch_size, self.kwargs["n_batches"], randomize=True, device=self.policy.device
		)
		losses = []
		for _ in range(self.kwargs["n_epochs"]):
			for batch in batches:
				loss = self._behaviour_cloning_update_weights(batch)
				losses.append(loss)
		self._last_policy = self.copy_policy(requires_grad=False)
		return float(np.mean(losses))

	def fit_buffer(
			self,
			buffer: ReplayBuffer,
	) -> float:
		"""
		Fit the agent on the given buffer.
		:param buffer: The replay buffer.
		:return: The loss.
		"""
		batch_size = min(len(buffer), self.kwargs["batch_size"])
		batches = buffer.get_batch_generator(
			batch_size, self.kwargs["n_batches"], randomize=True, device=self.policy.device
		)
		losses = []
		for _ in range(self.kwargs["n_epochs"]):
			for batch in batches:
				# predictions = self.policy.get_actions(batch.obs, as_numpy=False)
				# targets = self._last_policy.get_actions(batch.next_obs, as_numpy=False)
				# loss = self.update_weights(batch, predictions, targets)
				# self._last_policy.soft_update(self.policy, tau=self.kwargs["tau"])

				loss = self.update_policy_weights(batch)
				# self._last_policy = self._copy_policy(requires_grad=False)
				self._last_policy.soft_update(self.policy, tau=self.kwargs["tau"])
				losses.append(loss)
		self._last_policy = self.copy_policy(requires_grad=False)
		return float(np.mean(losses))

	def _behaviour_cloning_compute_continuous_loss(self, batch: BatchExperience, predictions) -> torch.Tensor:
		targets = batch.continuous_actions
		if torch.numel(batch.continuous_actions) == 0:
			continuous_loss = 0.0
		else:
			continuous_loss = self.continuous_criterion(predictions, targets.to(self.policy.device))
		return continuous_loss

	def _behaviour_cloning_compute_discrete_loss(self, batch: BatchExperience, predictions) -> torch.Tensor:
		targets = batch.discrete_actions
		if torch.numel(batch.discrete_actions) == 0:
			discrete_loss = 0.0
		else:
			warnings.warn("Discrete loss is not implemented with cross entropy loss. This is a temporary solution.")
			discrete_loss = self.discrete_criterion(predictions, targets.to(self.policy.device))
		return discrete_loss

	def _behaviour_cloning_update_weights(
			self,
			batch: BatchExperience,
	) -> float:
		assert torch.numel(batch.continuous_actions) + torch.numel(batch.discrete_actions) > 0
		predictions = self.policy.get_actions(batch.obs)
		bc_continuous_loss = self._behaviour_cloning_compute_continuous_loss(batch, predictions.continuous)
		bc_discrete_loss = self._behaviour_cloning_compute_discrete_loss(batch, predictions.discrete)
		bc_loss = bc_continuous_loss + bc_discrete_loss
		# Perform the backpropagation
		self.cloning_optimizer.zero_grad()
		bc_loss.backward()
		self.cloning_optimizer.step()
		return bc_loss.detach().cpu().numpy().item()
	
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

	def close(self):
		self.env.close()
