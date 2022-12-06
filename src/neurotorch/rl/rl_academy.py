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

	def __init__(self, buffer: Optional[ReplayBuffer] = None, **kwargs):
		self.buffer = buffer if buffer is not None else ReplayBuffer()
		self.trajectories: Dict[int, Trajectory] = defaultdict(Trajectory)
		self.cumulative_rewards: Dict[int, float] = defaultdict(lambda: 0.0)
		self._terminal_counter = 0
		self.min_rewards = kwargs.get("min_rewards", float('inf'))
		self.max_rewards = kwargs.get("max_rewards", float('-inf'))
		self.normalize_rewards = kwargs.get("normalize_rewards", False)

	@property
	def terminals_count(self) -> int:
		"""
		:return: The number of terminal steps
		"""
		return self._terminal_counter
	
	@property
	def max_abs_rewards(self) -> float:
		"""
		:return: The maximum absolute reward
		"""
		return max(abs(self.min_rewards), abs(self.max_rewards))
	
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
		self.min_rewards = min(self.min_rewards, np.min(rewards))
		self.max_rewards = max(self.max_rewards, np.max(rewards))
		if self.normalize_rewards:
			rewards = rewards / (self.max_abs_rewards + 1e-8)
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
		self._agents_history_maps_meta = {}
	
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
		kwargs.setdefault("normalize_rewards", False)

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
	
	def copy_agent(self, requires_grad: bool = False) -> Agent:
		"""
		Copy the agent to a new instance.
		
		:return: The copied agent.
		"""
		agent_copy = Agent.copy_from_agent(self.agent, requires_grad=requires_grad)
		agent_copy.policy.eval()
		return agent_copy

	def _update_optimizer_(self):
		self.policy_optimizer = torch.optim.Adam(
			self.policy.parameters(),
			lr=self.kwargs["init_lr"],
			weight_decay=self.kwargs["weight_decay"]
		)
		
	def _update_agents_history_maps_meta(self, agents_history_maps: AgentsHistoryMaps):
		self._agents_history_maps_meta = {
			"min_rewards": agents_history_maps.min_rewards,
			"max_rewards": agents_history_maps.max_rewards,
		}

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
		agents_history_maps = AgentsHistoryMaps(
			buffer, normalize_rewards=self.kwargs["normalize_rewards"], **self._agents_history_maps_meta
		)
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
		self._update_agents_history_maps_meta(agents_history_maps)
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

	def close(self):
		self.env.close()
