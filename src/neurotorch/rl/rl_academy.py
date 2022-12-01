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
from tqdm.auto import tqdm

from .agent import Agent
from .buffers import ReplayBuffer, Trajectory, Experience, BatchExperience
from .. import Trainer, LoadCheckpointMode
from ..callbacks.base_callback import BaseCallback, CallbacksList
from ..learning_algorithms.learning_algorithm import LearningAlgorithm
from ..utils import linear_decay


class AgentsHistoryMaps:
	r"""
	Class to store the mapping between agents and their history maps

	Attributes:
		trajectories (Dict[int, Trajectory]): Mapping between agent ids and their trajectories
		last_obs (Dict[int, np.ndarray]): Mapping between agent ids and their last observations
		last_action (Dict[int, np.ndarray]): Mapping between agent ids and their last actions
		cumulative_reward (Dict[int, float]): Mapping between agent ids and their cumulative rewards
	"""

	def __init__(self, buffer: Optional[ReplayBuffer] = None):
		self.buffer = buffer if buffer is not None else ReplayBuffer()
		self.trajectories: Dict[int, Trajectory] = defaultdict(Trajectory)
		self.last_obs: Dict[int, Any] = defaultdict()
		self.last_action: Dict[int, Any] = defaultdict()
		self.cumulative_reward: Dict[int, float] = defaultdict(lambda: 0.0)
		self._terminal_counter = 0

	@property
	def terminals_count(self) -> int:
		"""
		:return: The number of terminal steps
		"""
		return self._terminal_counter

	def update_terminals_(
			self,
			terminal_steps: TerminalSteps
	) -> List[float]:
		"""
		Execute terminal steps and return the rewards.
		
		:param terminal_steps: The terminal steps.
		
		:return: The rewards
		"""
		cumulative_rewards = []
		# For all Agents with a Terminal Step:
		for agent_id_terminated in terminal_steps:
			# Create its last experience (is last because the Agent terminated)
			last_experience = Experience(
				obs=deepcopy(self.last_obs[agent_id_terminated]),
				reward=terminal_steps[agent_id_terminated].reward,
				terminal=not terminal_steps[agent_id_terminated].interrupted,
				action=self.last_action[agent_id_terminated].copy(),
				next_obs=terminal_steps[agent_id_terminated].obs,
			)
			self.trajectories[agent_id_terminated].append_and_terminate(last_experience)
			# Clear its last observation and action (Since the trajectory is over)
			self.last_obs.pop(agent_id_terminated)
			self.last_action.pop(agent_id_terminated)
			# Report the cumulative reward
			cumulative_rewards.append(
				self.cumulative_reward.pop(agent_id_terminated, 0.0)
				+ terminal_steps[agent_id_terminated].reward
			)
			# Add the Trajectory to the buffer
			self.buffer.extend(self.trajectories.pop(agent_id_terminated))
			self._terminal_counter += 1
		return cumulative_rewards

	def update_decisions_(self, decision_steps: DecisionSteps):
		"""
		Execute the decision steps of the agents
		:param decision_steps: The decision steps
		:return: None
		"""
		# For all Agents with a Decision Step:
		for agent_id_decisions in decision_steps:
			# If the Agent requesting a decision has a "last observation"
			if agent_id_decisions in self.last_obs:
				# Create an Experience from the last observation and the Decision Step
				exp = Experience(
					obs=deepcopy(self.last_obs[agent_id_decisions]),
					reward=decision_steps[agent_id_decisions].reward,
					terminal=False,
					action=self.last_action[agent_id_decisions].copy(),
					next_obs=decision_steps[agent_id_decisions].obs,
				)
				# Update the Trajectory of the Agent and its cumulative reward
				self.trajectories[agent_id_decisions].append(exp)
				self.cumulative_reward[agent_id_decisions] += decision_steps[agent_id_decisions].reward
			# Store the observation as the new "last observation"
			self.last_obs[agent_id_decisions] = decision_steps[agent_id_decisions].obs

	def update_actions_(self, actions: TensorActionTuple, decision_steps: DecisionSteps):
		"""
		Add the actions to the last observation of the agents.
		:param actions: The actions
		:param decision_steps: The decision steps
		:return: None
		"""
		actions_list = unbatch_actions(actions)
		for agent_index, agent_id in enumerate(decision_steps.agent_id):
			self.last_action[agent_id] = actions_list[agent_index]

	def update_(
			self,
			decision_steps: Optional[DecisionSteps] = None,
			terminal_steps: Optional[TerminalSteps] = None,
			actions: Optional[TensorActionTuple] = None,
	) -> Optional[List[float]]:
		"""
		Update the replay buffer with the given steps.
		
		:param decision_steps: The decision steps
		:param terminal_steps: The terminal steps
		:param actions: The actions
		:return: The cumulative rewards if the terminal steps are given
		"""
		cumulative_rewards = None
		if terminal_steps is not None:
			cumulative_rewards = self.update_terminals_(terminal_steps)
		if decision_steps is not None:
			self.update_decisions_(decision_steps)
		if actions is not None:
			assert decision_steps is not None, "If actions are given, decision_steps must be given"
			self.update_actions_(actions, decision_steps)
		return cumulative_rewards


class RLAcademy(Trainer):
	def __init__(
			self,
			agent: Agent,
			*,
			predict_method: str = "get_actions",
			learning_algorithm: Optional[LearningAlgorithm] = None,
			callbacks: Optional[Union[List[BaseCallback], CallbacksList, BaseCallback]] = None,
			verbose: bool = True,
			**kwargs
	):
		"""
		Initialize the RL Academy.
		:param env: The environment to train the agent in.
		:param agent:
		:param behavior_name:
		:param checkpoint_folder:
		:param curriculum:
		:param kwargs:
		"""
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
		kwargs.setdefault("n_epochs", 3)
		kwargs.setdefault("init_lr", 3.0e-4)
		kwargs.setdefault("min_lr", 3.0e-4)
		kwargs.setdefault("lr", kwargs["init_lr"])
		kwargs.setdefault("weight_decay", 1e-5)
		kwargs.setdefault("init_epsilon", 0.01)
		kwargs.setdefault("epsilon_decay", 0.995)
		kwargs.setdefault("min_epsilon", 0.0)
		kwargs.setdefault("gamma", 0.99)
		kwargs.setdefault("n_batches", 3)
		kwargs.setdefault("tau", 1/kwargs["n_batches"])
		kwargs.setdefault("batch_size", 256)
		kwargs.setdefault("update_freq", 32)
		kwargs.setdefault("bc_strength", 0.5)
		kwargs.setdefault("buffer_size", 4096)
		kwargs.setdefault("clip_ratio", 0.2)
		kwargs.setdefault("use_priority_buffer", True)

		assert kwargs["batch_size"] <= kwargs["buffer_size"]
		assert kwargs["batch_size"] > 0
		return kwargs

	def _copy_policy(self, requires_grad: bool = False) -> Agent:
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
			n_trajectories: int,
			buffer: Optional[ReplayBuffer] = None,
			epsilon: float = 0.0,
			p_bar_position: int = 0,
			verbose: Optional[bool] = None
	) -> Tuple[ReplayBuffer, List[float]]:
		if buffer is None:
			buffer = ReplayBuffer(self.kwargs["buffer_size"], use_priority=self.kwargs["use_priority_buffer"])
		if verbose is None:
			verbose = self.verbose
		agents_history_maps = AgentsHistoryMaps(buffer)
		cumulative_rewards: List[float] = []
		p_bar = tqdm(
			total=n_trajectories, disable=not verbose, desc="Generating Trajectories", position=p_bar_position
		)
		observations = self.env.reset()
		while agents_history_maps.terminals_count < n_trajectories:  # While not enough data in the buffer
			decision_steps, terminal_steps = self.env.step()
			actions = None
			if len(decision_steps) > 0:
				if np.random.random() < epsilon:
					actions = self.agent.get_random_actions(len(decision_steps))
				else:
					actions = self.agent.get_actions(decision_steps.obs)
				self.env.set_actions(self.behavior_name, actions.to_numpy())
				observations, rewards, dones, infos = self.env.step(actions)
			new_cumulative_rewards = agents_history_maps.update_(decision_steps, terminal_steps, actions)
			p_bar.update(min(len(new_cumulative_rewards), n_trajectories - len(cumulative_rewards)))
			cumulative_rewards.extend(new_cumulative_rewards)
			p_bar.set_postfix(cumulative_reward=f"{np.mean(cumulative_rewards) if cumulative_rewards else 0.0:.3f}")
			self.env._update_k_p_on_datum()
		p_bar.close()
		return buffer, cumulative_rewards

	def load_checkpoint(
			self,
			load_checkpoint_mode: LoadCheckpointMode = LoadCheckpointMode.BEST_ITR
	) -> dict:
		checkpoint = self.checkpoint_manager.load_checkpoint(load_checkpoint_mode)
		self.policy.load_state_dict(checkpoint[CheckpointManager.CHECKPOINT_STATE_DICT_KEY], strict=True)
		return checkpoint

	def plot_training_history(self, training_history: TrainingHistory = None, show: bool = False) -> str:
		if training_history is None:
			training_history = self.training_histories
		save_path = f"./{self.checkpoint_folder}/training_history.png"
		os.makedirs(f"./{self.checkpoint_folder}/", exist_ok=True)
		training_history.plot(save_path=save_path, show=show)
		return save_path

	def check_and_load_state_from_academy_checkpoint(
			self,
			load_checkpoint_mode: LoadCheckpointMode = None,
			force_overwrite: bool = False,
	) -> int:
		start_itr = 0
		if load_checkpoint_mode is None:
			if os.path.exists(self.checkpoint_manager.checkpoints_meta_path):
				if force_overwrite:
					shutil.rmtree(self.checkpoint_folder)
				else:
					raise ValueError(
						f"{self.checkpoint_manager.checkpoints_meta_path} already exists. "
						f"Set force_overwrite flag to True to overwrite existing saves."
					)
		else:
			try:
				checkpoint = self.load_checkpoint(load_checkpoint_mode)
				self.policy.load_state_dict(checkpoint[CheckpointManager.CHECKPOINT_STATE_DICT_KEY], strict=True)
				self.policy_optimizer.load_state_dict(checkpoint[CheckpointManager.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY])
				start_itr = int(checkpoint[CheckpointManager.CHECKPOINT_ITR_KEY]) + 1
				self.training_histories: TrainingHistoriesMap = checkpoint[CheckpointManager.CHECKPOINT_TRAINING_HISTORY_KEY]
				temp_curriculum = self.curriculum
				self.curriculum = self.training_histories.curriculum
				if temp_curriculum is not None:
					self.curriculum.update_teachers_and_channels(temp_curriculum)
				self.plot_training_history(show=False)
			except FileNotFoundError as e:
				if self.verbose:
					warnings.warn(f"Error: {e}", Warning)
					warnings.warn("No such checkpoint. Fit from beginning.")
		return start_itr

	def _init_train_buffer(self) -> ReplayBuffer:
		self.env.reset()
		if self.curriculum is not None:
			self.curriculum.on_iteration_start()
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
	):
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
			env = self.current_training_state.objects["env"]
			env.reset()
			itr_loss = self._exec_iteration(env)
			self.update_itr_metrics_state_(**itr_loss)
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
			**kwargs,
	) -> Dict[str, float]:
		with torch.no_grad():
			torch.cuda.empty_cache()
		losses = {}
		
		self.model.train()
		self.callbacks.on_train_begin(self)
		self.update_state_(batch_is_train=True)
		train_losses = []
		for epoch_idx in range(self.current_training_state.n_epochs):
			self.update_state_(epoch=epoch_idx)
			train_losses.append(self._exec_epoch(train_dataloader))
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

	def _make_itr_checkpoint(self, itr: int, itr_metrics: Dict[str, float], best_saved_rewards: float) -> float:
		is_best = itr_metrics["Rewards"] > best_saved_rewards
		if is_best:
			best_saved_rewards = itr_metrics["Rewards"]
		self.checkpoint_manager.save_checkpoint(
			itr, itr_metrics, is_best,
			state_dict=self.policy.state_dict(),
			optimizer_state_dict=self.policy_optimizer.state_dict(),
			training_history=self.training_histories,
		)
		self.plot_training_history(show=False)
		return best_saved_rewards

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
		self._last_policy = self._copy_policy(requires_grad=False)
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
		self._last_policy = self._copy_policy(requires_grad=False)
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

	def close(self):
		self.env.close()
