import pickle
from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable, List, NamedTuple, Optional, Dict, Iterator

import numpy as np
import torch
from queue import PriorityQueue

from ..transforms.base import to_tensor, ToDevice, to_numpy


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
            rewards_horizon: Optional[List[float]] = None,
            others: Optional[dict] = None
    ):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.terminal = terminal
        self.next_obs = next_obs
        self.rewards_horizon = rewards_horizon or []
        self._discounted_reward = discounted_reward
        self._advantage = advantage
        self.others = others or {}

    @property
    def observation(self):
        return self.obs

    @property
    def metrics(self):
        return self.others

    @metrics.setter
    def metrics(self, metrics):
        self.others = metrics

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
            device: torch.device = torch.device("cpu"),
    ):
        """
        An object that contains a batch of experiences as tensors.

        :param batch: A list of Experience objects.
        :param device: The device to use for the tensors.
        """
        batch = deepcopy(batch)
        self._batch = batch
        self._device = device
        self._to = ToDevice(device=device)

        self.obs: List[torch.Tensor] = self._make_obs_batch(batch)
        self.rewards: torch.Tensor = self._make_rewards_batch(batch)
        self.terminals: torch.Tensor = self._make_terminals_batch(batch)
        self.actions = self._make_actions_batch(batch)
        self.next_obs: List[torch.Tensor] = self._make_next_obs_batch(batch)
        self.others: List[dict] = [ex.others for ex in batch]

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

    def __getitem__(self, item):
        if isinstance(item, slice):
            return BatchExperience(batch=self._batch[item], device=self.device)
        return BatchExperience(batch=[self._batch[item]], device=self.device)

    def _make_obs_batch(self, batch: List[Experience]) -> List[torch.Tensor]:
        as_dict = isinstance(batch[0].obs, dict)
        if as_dict:
            obs = {
                key: torch.stack([to_tensor(ex.obs[key]) for ex in batch])
                for key in batch[0].obs
            }
            return self._to(obs)
        return self._to(torch.stack([to_tensor(ex.obs) for ex in batch]))

    def _make_next_obs_batch(self, batch: List[Experience]) -> List[torch.Tensor]:
        as_dict = isinstance(batch[0].next_obs, dict)
        if as_dict:
            obs = {
                key: torch.stack([to_tensor(ex.next_obs[key]) for ex in batch])
                for key in batch[0].next_obs
            }
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
            action = {
                key: torch.stack([to_tensor(ex.action[key]) for ex in batch])
                for key in batch[0].action
            }
            return self._to(action)
        return self._to(torch.stack([to_tensor(ex.action) for ex in batch]))

    def _make_rewards_horizon_batch(self, batch: List[Experience]):
        return [self._to(to_tensor(ex.rewards_horizon)) for ex in batch]


class Trajectory:
    """
    A trajectory is a list of experiences.
    """
    def __init__(
            self,
            experiences: Optional[List[Experience]] = None,
            gamma: Optional[float] = None,
            **kwargs,
    ):
        self.experiences = experiences if experiences is not None else []
        self._propagated_flag = False
        self.gamma = gamma
        self.rewards_horizon = kwargs.get("rewards_horizon", 1)
        assert self.rewards_horizon > 0, "The rewards horizon must be greater than 0."

    @property
    def terminated(self):
        return self.experiences[-1].terminal if self.experiences else False

    @property
    def terminal(self):
        return self.terminated

    @property
    def _default_gamma(self):
        return self.gamma if self.gamma is not None else 0.99

    @property
    def cumulative_reward(self):
        return sum([exp.reward for i, exp in enumerate(self.experiences)])

    @property
    def terminal_reward(self):
        return self.experiences[-1].reward

    @property
    def propagated(self):
        return self._propagated_flag

    def is_empty(self):
        return len(self) == 0

    def propagate(self):
        self.propagate_rewards()
        self.make_rewards_horizon()
        self._propagated_flag = True

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

    def make_rewards_horizon(self):
        for i in range(len(self.experiences)):
            self.experiences[i].rewards_horizon = [self.experiences[i].reward]
            for j in range(i, min(i + self.rewards_horizon, len(self.experiences))):
                self.experiences[i].rewards_horizon.append(self.experiences[j].reward)

    def compute_horizon_rewards(self):
        raise NotImplementedError()

    def propagate_values(self, lmbda: Optional[float] = 0.95):
        raise NotImplementedError("Not implemented yet.")

    def __iter__(self):
        return iter(self.experiences)

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, index: int) -> Experience:
        return self.experiences[index]

    def append(self, experience: Experience):
        self.experiences.append(experience)
        self._propagated_flag = False
        if experience.terminal:
            self.propagate()

    def append_and_propagate(self, experience: Experience):
        self.append(experience)
        self.propagate()

    def update_others(self, others_list: List[dict]):
        assert len(others_list) == len(self.experiences), "The number of experiences must be the same."
        for i, others in enumerate(others_list):
            self.experiences[i].others.update(others)


class ReplayBuffer:
    def __init__(self, capacity=np.inf, seed=None, use_priority=False, **kwargs):
        self.__capacity = capacity
        self._seed = seed
        self.random_generator = np.random.RandomState(seed)
        self.data: List[Experience] = []
        self._counter = 0
        self._counter_is_started = False
        self.use_priority = use_priority
        self.priority_key = kwargs.get("priority_key", "discounted_reward")

    @property
    def counter(self):
        return self._counter

    @property
    def capacity(self):
        return self.__capacity

    @property
    def full(self):
        return len(self) >= self.capacity

    @property
    def empty(self):
        return len(self) == 0

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

    def extend(self, iterable: Iterable[Experience]) -> 'ReplayBuffer':
        _ = [self.store(e) for e in iterable]
        return self

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx: int) -> Experience:
        return self.data[idx]

    def store(self, element: Experience) -> 'ReplayBuffer':
        """
        Stores an element. If the replay buffer is already full, deletes the oldest
        element to make space.
        """
        if len(self.data) >= self.__capacity:
            if self.use_priority:
                self.data.pop(np.argmin([np.abs(getattr(e, self.priority_key, 0.0)) for e in self.data]))
            else:
                self.data.pop(0)
        self.data.append(deepcopy(element))
        if self._counter_is_started:
            self._counter += 1
        return self

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
    ) -> Iterator[BatchExperience]:
        """
        Returns a generator of batch_size elements from the buffer.
        """
        if batch_size > len(self.data) or batch_size <= 0:
            batch_size = len(self.data)
        max_idx = int(batch_size * int(len(self) / batch_size))
        indexes = np.arange(max_idx)
        if randomize:
            self.random_generator.shuffle(indexes)
        indexes = indexes.reshape(-1, batch_size)
        if n_batches is None or n_batches > len(indexes) or n_batches < 0:
            n_batches = indexes.shape[0]
        else:
            n_batches = min(n_batches, indexes.shape[0])
        for i in range(n_batches):
            batch = [self.data[j] for j in indexes[i]]
            yield BatchExperience(batch, device=device)

    def clear(self):
        self.data.clear()

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


def _re_zero(*args, **kwargs):
    """
    Use instead of 'lambda: 0.0' to avoid pickling issues.
    """
    return 0.0


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
        self.trajectories.update(kwargs.get('trajectories', {}))
        self.cumulative_rewards: Dict[int, list] = defaultdict(list)
        self.terminal_rewards: Dict[int, float] = defaultdict(_re_zero)
        self._terminal_counter = 0
        self._experience_counter = 0
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
    def experience_count(self) -> int:
        """
        :return: The number of experiences
        """
        return self._experience_counter

    @property
    def max_abs_rewards(self) -> float:
        """
        :return: The maximum absolute reward
        """
        return max(abs(self.min_rewards), abs(self.max_rewards))

    @property
    def cumulative_rewards_as_array(self) -> np.ndarray:
        """
        :return: The cumulative rewards as an array
        """
        cum_rewards_list = sum(self.cumulative_rewards.values(), [])
        # cum_rewards_list = []
        # for agent_id, rewards in self.cumulative_rewards.items():
        # 	cum_rewards_list.extend(rewards)
        return np.asarray(cum_rewards_list)

    @property
    def mean_cumulative_rewards(self) -> float:
        """
        :return: The mean cumulative rewards
        """
        cumulative_rewards = self.cumulative_rewards_as_array
        if cumulative_rewards.size == 0:
            return 0.0
        return np.nanmean(cumulative_rewards).item()

    def update_trajectories_(
            self,
            *,
            observations,
            actions,
            next_observations,
            rewards,
            terminals,
            truncated=None,
            infos=None,
            others=None,
    ) -> List[Trajectory]:
        """
        Updates the trajectories of the agents and returns the trajectories of the agents that have been terminated.

        :param observations: The observations
        :param actions: The actions
        :param next_observations: The next observations
        :param rewards: The rewards
        :param terminals: The terminals
        :param truncated: The truncated
        :param infos: The infos
        :param others: The others

        :return: The terminated trajectories.
        """
        from .utils import get_item_from_batch

        actions = deepcopy(to_numpy(actions))
        observations, next_observations = deepcopy(to_numpy(observations)), deepcopy(to_numpy(next_observations))
        rewards, terminals = deepcopy(to_numpy(rewards)), deepcopy(to_numpy(terminals))
        if others is None:
            others = [None] * len(observations)
        self.min_rewards = min(self.min_rewards, np.min(rewards))
        self.max_rewards = max(self.max_rewards, np.max(rewards))
        if self.normalize_rewards:
            rewards = rewards / (self.max_abs_rewards + 1e-8)

        finished_trajectories = []
        for i in range(len(terminals)):
            if self.trajectories[i].terminated:
                continue
            if terminals[i]:
                self.trajectories[i].append_and_propagate(
                    Experience(
                        obs=get_item_from_batch(observations, i),
                        reward=rewards[i],
                        terminal=terminals[i],
                        action=get_item_from_batch(actions, i),
                        next_obs=get_item_from_batch(next_observations, i),
                        others=get_item_from_batch(others, i),
                    )
                )
                self.cumulative_rewards[i].append(self.trajectories[i].cumulative_reward)
                self.terminal_rewards[i] = self.trajectories[i].terminal_reward
                finished_trajectory = self.trajectories.pop(i)
                finished_trajectories.append(finished_trajectory)
                # self.buffer.extend(finished_trajectory)
                self._terminal_counter += 1
                self._experience_counter += 1
            else:
                self.trajectories[i].append(
                    Experience(
                        obs=get_item_from_batch(observations, i),
                        reward=rewards[i],
                        terminal=terminals[i],
                        action=get_item_from_batch(actions, i),
                        next_obs=get_item_from_batch(next_observations, i),
                        others=get_item_from_batch(others, i),
                    )
                )
                self._experience_counter += 1
        return finished_trajectories

    def propagate_all(self) -> List[Trajectory]:
        """
        Propagate all the trajectories and return the finished ones.

        :return: All the trajectories.
        :rtype: List[Trajectory]
        """
        trajectories = []
        for i in range(len(self.trajectories)):
            if not self.trajectories[i].propagated:
                self.trajectories[i].propagate()
            if self.trajectories[i].terminated:
                self.cumulative_rewards[i].append(self.trajectories[i].cumulative_reward)
                trajectory = self.trajectories.pop(i)
                trajectories.append(trajectory)
                self._terminal_counter += 1
            else:
                trajectory = self.trajectories[i]
        # self.buffer.extend(trajectory)
        return trajectories

    def propagate_and_get_all(self) -> List[Trajectory]:
        """
        Propagate all the trajectories and return all the trajectories.

        :return: All the trajectories
        :rtype: List[Trajectory]
        """
        trajectories = []
        for i in range(len(self.trajectories)):
            if not self.trajectories[i].propagated:
                self.trajectories[i].propagate()
            if self.trajectories[i].terminated:
                self.cumulative_rewards[i].append(self.trajectories[i].cumulative_reward)
                trajectory = self.trajectories.pop(i)
                self._terminal_counter += 1
            else:
                trajectory = self.trajectories[i]
            trajectories.append(trajectory)
        # self.buffer.extend(trajectory)
        return trajectories

    def clear(self) -> List[Trajectory]:
        trajectories = self.propagate_and_get_all()
        self.trajectories.clear()
        self.cumulative_rewards.clear()
        self._terminal_counter = 0
        self._experience_counter = 0
        return trajectories
