import pickle
from copy import deepcopy
from typing import Any, Iterable, List, NamedTuple, Optional

import numpy as np
import torch
from queue import PriorityQueue

from ..transforms.base import to_tensor, ToDevice


class Experience:
	"""
	An experience contains the data of one Agent transition.
	- Observation
	- Action
	- Reward
	- Terminal flag
	- Next Observation
	"""
	def __init__(
			self,
			obs: Any,
			action: Any,
			reward: float,
			terminal: bool,
			next_obs: Any,
			discounted_reward: Optional[float] = None,
			advantage: Optional[float] = None,
	):
		self.obs = obs
		self.action = action
		self.reward = reward
		self.terminal = terminal
		self.next_obs = next_obs
		self._discounted_reward = discounted_reward
		self._advantage = advantage

	@property
	def discounted_reward(self) -> float:
		if self._discounted_reward is None:
			return self.reward
		return self._discounted_reward

	@discounted_reward.setter
	def discounted_reward(self, value: float):
		self._discounted_reward = value

	@property
	def advantage(self) -> float:
		if self._advantage is None:
			return self.discounted_reward
		return self._advantage

	@advantage.setter
	def advantage(self, value: float):
		self._advantage = value


class BatchExperience:
	def __init__(
			self,
			batch: List[Experience],
			device: torch.device("cpu"),
	):
		"""
		An object that contains a batch of experiences as tensors.

		:param batch: A list of Experience objects.
		:param device: The device to use for the tensors.
		"""
		self._device = device
		self._to = ToDevice(device=device)
		# self._nb_obs = len(batch[0].obs)

		self.obs: List[torch.Tensor] = self._make_obs_batch(batch)
		self.rewards: torch.Tensor = self._make_rewards_batch(batch)
		self.terminals: torch.Tensor = self._make_terminals_batch(batch)
		self.actions = self._make_actions_batch(batch)
		self.next_obs: List[torch.Tensor] = self._make_next_obs_batch(batch)
		self.discounted_rewards: torch.Tensor = self._make_discounted_rewards_batch(batch)
		self.advantages: torch.Tensor = self._make_advantages_batch(batch)

	@property
	def device(self):
		return self._device

	@device.setter
	def device(self, value):
		self._device = value
		# self.obs = [obs.to(value) for obs in self.obs]
		# self.next_obs = [next_obs.to(value) for next_obs in self.next_obs]
		# self.rewards = self.rewards.to(value)
		# self.terminals = self.terminals.to(value)
		# self.continuous_actions = self.continuous_actions.to(value)
		# self.discrete_actions = self.discrete_actions.to(value)
		# self.discounted_rewards = self.discounted_rewards.to(value)
		# self.advantages = self.advantages.to(value)
		self._to.device = value

	def __len__(self):
		return self.rewards.shape[0]

	def _make_obs_batch(self, batch: List[Experience]) -> List[torch.Tensor]:
		as_dict = isinstance(batch[0].obs, dict)
		if as_dict:
			obs = deepcopy(batch[0].obs)
			for key in obs:
				obs[key] = torch.stack([to_tensor(ex.obs[key]) for ex in batch])
			return self._to(obs)
		return self._to(torch.stack([to_tensor(ex.obs) for ex in batch]))

	def _make_next_obs_batch(self, batch: List[Experience]) -> List[torch.Tensor]:
		as_dict = isinstance(batch[0].next_obs, dict)
		if as_dict:
			obs = deepcopy(batch[0].next_obs)
			for key in obs:
				obs[key] = torch.stack([to_tensor(ex.next_obs[key]) for ex in batch])
			return self._to(obs)
		return self._to(torch.stack([to_tensor(ex.next_obs) for ex in batch]))

	def _make_rewards_batch(self, batch: List[Experience]) -> torch.Tensor:
		return self._to(torch.stack([to_tensor(ex.reward) for ex in batch]))

	def _make_terminals_batch(self, batch: List[Experience]) -> torch.Tensor:
		return self._to(torch.stack([to_tensor(ex.terminal) for ex in batch]))

	def _make_discounted_rewards_batch(self, batch: List[Experience]) -> torch.Tensor:
		return self._to(torch.stack([to_tensor(ex.discounted_reward) for ex in batch]))

	def _make_advantages_batch(self, batch: List[Experience]) -> torch.Tensor:
		return self._to(torch.stack([to_tensor(ex.advantage) for ex in batch]))
	
	def _make_actions_batch(self, batch: List[Experience]) -> torch.Tensor:
		as_dict = isinstance(batch[0].action, dict)
		if as_dict:
			action = deepcopy(batch[0].action)
			for key in action:
				action[key] = torch.stack([to_tensor(ex.action[key]) for ex in batch])
			return self._to(action)
		return self._to(torch.stack([to_tensor(ex.action) for ex in batch]))


class Trajectory:
	"""
	A trajectory is a list of experiences.
	"""
	def __init__(
			self,
			experiences: Optional[List[Experience]] = None,
			gamma: Optional[float] = None,
	):
		self.experiences = experiences if experiences is not None else []
		self._terminal_flag = experiences[-1].terminal if experiences else False
		self.gamma = gamma
	
	@property
	def terminated(self):
		return self._terminal_flag

	@property
	def _default_gamma(self):
		return self.gamma if self.gamma is not None else 0.99
	
	@property
	def cumulative_reward(self):
		return sum([exp.reward for i, exp in enumerate(self.experiences)])

	def set_terminal(self, terminal: bool):
		self._terminal_flag = terminal
		if self._terminal_flag:
			self.propagate_rewards()
			
	def terminate(self):
		self.set_terminal(True)

	def propagate_rewards(self, gamma: Optional[float] = 0.99):
		"""
		Propagate the rewards to the next experiences.
		"""
		gamma = gamma if gamma is not None else self._default_gamma
		for i in reversed(range(len(self.experiences))):
			if i == len(self.experiences) - 1:
				self.experiences[i].discounted_reward = self.experiences[i].reward
			else:
				self.experiences[i].discounted_reward = (
						self.experiences[i].reward + gamma * self.experiences[i + 1].discounted_reward
				)

	def propagate_values(self, lmbda: Optional[float] = 0.95):
		raise NotImplementedError("Not implemented yet.")

	def __iter__(self):
		return iter(self.experiences)

	def __len__(self):
		return len(self.experiences)

	def __getitem__(self, index: int) -> Experience:
		return self.experiences[index]

	def append(self, experience: Experience):
		if self._terminal_flag:
			raise ValueError("Cannot append experience to a terminal trajectory.")
		self.experiences.append(experience)
		self.set_terminal(experience.terminal)

	def append_and_terminate(self, experience: Experience):
		self.append(experience)
		self.set_terminal(True)


class ReplayBuffer:
	def __init__(self, capacity=np.inf, seed=None, use_priority=True):
		self.__capacity = capacity
		self._seed = seed
		self.random_generator = np.random.RandomState(seed)
		self.data: List[Experience] = []
		self._counter = 0
		self._counter_is_started = False
		self.use_priority = use_priority

	@property
	def counter(self):
		return self._counter

	@property
	def capacity(self):
		return self.__capacity
	
	def __str__(self):
		_repr = f"ReplayBuffer("
		_repr += f"capacity={self.capacity}"
		_repr += f", size={len(self)}"
		_repr += f", use_priority={self.use_priority}"
		_repr += f", seed={self._seed}"
		_repr += f")"
		return _repr
	
	def set_seed(self, seed: int):
		self.random_generator.seed(seed)

	def start_counter(self):
		self._counter_is_started = True
		self._counter = 0

	def stop_counter(self):
		self._counter_is_started = False
		self._counter = 0

	def reset_counter(self):
		self.stop_counter()

	def increment_counter(self, increment: int = 1):
		self._counter += increment

	def increase_capacity(self, increment: int):
		self.__capacity += increment
	
	def extend(self, iterable: Iterable[Experience]):
		_ = [self.store(e) for e in iterable]
	
	def __len__(self):
		return len(self.data)

	def __iter__(self):
		return iter(self.data)

	def __getitem__(self, idx: int) -> Experience:
		return self.data[idx]
	
	def store(self, element: Experience):
		"""
		Stores an element. If the replay buffer is already full, deletes the oldest
		element to make space.
		"""
		if len(self.data) >= self.__capacity:
			if self.use_priority:
				self.data.pop(np.argmin([np.abs(e.advantage) for e in self.data]))
			else:
				self.data.pop(0)
		self.data.append(element)
		if self._counter_is_started:
			self._counter += 1
	
	def get_random_batch(self, batch_size: int) -> List[Experience]:
		"""
		Returns a list of batch_size elements from the buffer.
		"""
		return self.random_generator.choice(self.data, size=batch_size)
	
	def get_batch_tensor(self, batch_size: int, device='cpu') -> BatchExperience:
		"""
		Returns a list of batch_size elements from the buffer.
		"""
		batch = self.get_random_batch(batch_size)
		return BatchExperience(batch, device=device)
	
	def get_batch_generator(
			self,
			batch_size: int,
			n_batches: int = None,
			randomize: bool = True,
			device='cpu',
	) -> Iterable[BatchExperience]:
		"""
		Returns a generator of batch_size elements from the buffer.
		"""
		max_idx = int(batch_size * int(len(self) / batch_size))
		indexes = np.arange(max_idx).reshape(-1, batch_size)
		if n_batches is None:
			n_batches = indexes.shape[0]
		else:
			n_batches = min(n_batches, indexes.shape[0])
		if randomize:
			self.random_generator.shuffle(indexes)
		for i in range(n_batches):
			batch = [self.data[j] for j in indexes[i]]
			yield BatchExperience(batch, device=device)

	def save(self, filename: str):
		buffer_copy = deepcopy(self)
		for i, e in enumerate(buffer_copy.data):
			buffer_copy.data[i] = Experience(
				obs=e.obs,
				action=e.action,
				reward=e.reward,
				terminal=e.terminal,
				next_obs=e.next_obs,
			)
		with open(filename, 'wb') as file:
			pickle.dump(self, file)

	@staticmethod
	def load(filename: str) -> 'ReplayBuffer':
		with open(filename, 'rb') as file:
			buffer = pickle.load(file)
		for i, e in enumerate(buffer.data):
			buffer.data[i] = Experience(
				obs=e.obs,
				action=e.action,
				reward=e.reward,
				terminal=e.terminal,
				next_obs=e.next_obs,
			)
		return buffer
