from typing import Sequence, Union, Optional, Dict, Any, List

import numpy as np
import torch
import gym

from ..modules.base import BaseModel
from ..modules.sequential import Sequential
try:
	from ..modules.layers import Linear
except ImportError:
	from .utils import Linear, space_to_spec, obs_batch_to_sequence, space_to_continuous_shape
from .utils import obs_sequence_to_batch


class Agent:
	def __init__(
			self,
			observation_space: gym.spaces.Space,
			action_space: gym.spaces.Space,
			behavior_name: str,
			policy: Optional[BaseModel] = None,
			**kwargs
	):
		"""
		Constructor for BaseAgent class.

		:param policy: The model to use.
		:type policy: BaseModel
		"""
		super().__init__(**kwargs)
		self.kwargs = kwargs
		self.observation_space = observation_space
		self.action_space = action_space
		self.behavior_name = behavior_name
		self.policy = policy
		if self.policy is None:
			self.policy = self._create_default_policy()
	
	@property
	def observation_spec(self) -> Dict[str, Any]:
		return space_to_spec(self.observation_space)
	
	@property
	def action_spec(self) -> Dict[str, Any]:
		return space_to_spec(self.action_space)
	
	@property
	def discrete_actions(self) -> List[str]:
		return [k for k, v in self.action_spec.items() if isinstance(v, gym.spaces.Discrete)]
	
	@property
	def continuous_actions(self) -> List[str]:
		return [k for k, v in self.action_spec.items() if not isinstance(v, gym.spaces.Discrete)]
		
	def _create_default_policy(self) -> BaseModel:
		"""
		Create the default policy.

		:return: The default policy.
		:rtype: BaseModel
		"""
		default_policy = Sequential(layers=[
			{
				k: Linear(
					input_size=int(space_to_continuous_shape(v, flatten_spaces=True)[0]),
					output_size=self.kwargs.get("default_hidden_units", 256),
					activation="ReLu"
				)
				for k, v in self.observation_spec.items()
			},
			*[
				Linear(
					input_size=self.kwargs.get("default_hidden_units", 256),
					output_size=self.kwargs.get("default_hidden_units", 256),
					activation="ReLu"
				)
				for _ in range(self.kwargs.get("default_hidden_layers", 1))
			],
			{
				k: Linear(
					input_size=self.kwargs.get("default_hidden_units", 256),
					output_size=int(space_to_continuous_shape(v, flatten_spaces=True)[0]),
					activation="ReLu"
				)
				for k, v in self.action_spec.items()
			}
		]).build()
		return default_policy

	def __call__(self, *args, **kwargs):
		"""
		Call the agent.

		:return: The output of the agent.
		"""
		return self.policy(*args, **kwargs)
	
	def get_actions(
			self,
			obs: Sequence[Union[np.ndarray, torch.Tensor, Dict[str, Union[np.ndarray, torch.Tensor]]]],
			**kwargs
	) -> Any:
		"""
		Get the actions for the given observations.
		
		:param obs: The observations.
		:type obs: Sequence[Union[np.ndarray, torch.Tensor, Dict[str, Union[np.ndarray, torch.Tensor]]]]
		:param kwargs: Keywords arguments.
		
		:keyword str re_format: The format to reformat the discrete actions to. Default is "index" which
			will return the index of the action. For other options see :mth:`format_batch_discrete_actions`.
		:keyword bool as_numpy: Whether to return the actions as numpy arrays. Default is True.
		
		:return: The actions.
		"""
		as_batch = kwargs.get("as_batch", True)  # TODO: if as_batch is False, then return a single action
		re_as_dict = kwargs.get("re_as_dict", isinstance(obs[0], dict))
		re_format = kwargs.get("re_format", "index")
		as_numpy = kwargs.get("as_numpy", True)
		
		obs_as_batch = obs_sequence_to_batch(obs)
		actions_as_batch = self.policy(obs_as_batch, **kwargs)
		actions_as_batch_fmt = self.format_batch_discrete_actions(actions_as_batch, re_format=re_format)
		actions_as_seq = obs_batch_to_sequence(actions_as_batch_fmt, as_numpy=as_numpy)
		if not re_as_dict:
			if not all([len(a) == 1 for a in actions_as_seq]):
				raise ValueError("Cannot re-assemble actions as sequence because they are not all of length 1.")
			actions_as_seq = [a[list(a.keys())[0]] for a in actions_as_seq]
		return actions_as_seq
	
	def format_batch_discrete_actions(
			self,
			actions: Union[torch.Tensor, Dict[str, torch.Tensor]],
			re_format: str = "logits",
			**kwargs
	) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
		"""
		Format the batch of actions. If actions is a dict, then it is assumed that the keys are the action names and the
		values are the actions. In this case, all the values where their keys are in `self.discrete_actions` will be
		formatted. If actions is a tensor, then the actions will be formatted if `self.discrete_actions` is not empty.
		
		:param actions: The actions.
		:param re_format: The format to reformat the actions to. Can be "logits", "probs", "index", or "one_hot".
		:param kwargs: Keywords arguments.
		:return: The formatted actions.
		"""
		discrete_actions = kwargs.get("discrete_actions", self.discrete_actions)
		if re_format.lower() == "logits":
			return actions
		elif re_format.lower() == "probs":
			if isinstance(actions, torch.Tensor):
				return torch.softmax(actions, dim=-1) if len(discrete_actions) >= 1 else actions
			elif isinstance(actions, dict):
				return {k: (torch.softmax(v, dim=-1) if k in discrete_actions else v) for k, v in actions.items()}
			else:
				raise ValueError(f"Cannot format actions of type {type(actions)}.")
		elif re_format.lower() == "index":
			if isinstance(actions, torch.Tensor):
				return torch.argmax(actions, dim=-1) if len(discrete_actions) >= 1 else actions
			elif isinstance(actions, dict):
				return {k: (torch.argmax(v, dim=-1) if k in discrete_actions else v) for k, v in actions.items()}
			else:
				raise ValueError(f"Cannot format actions of type {type(actions)}.")
		elif re_format.lower() == "one_hot":
			if isinstance(actions, torch.Tensor):
				return (
					torch.nn.functional.one_hot(torch.argmax(actions, dim=-1), num_classes=actions.shape[-1])
					if len(discrete_actions) >= 1 else actions
				)
			elif isinstance(actions, dict):
				return {
					k: (
						torch.nn.functional.one_hot(torch.argmax(v, dim=-1), num_classes=v.shape[-1])
						if k in discrete_actions else v
					)
					for k, v in actions.items()
				}
			else:
				raise ValueError(f"Cannot format actions of type {type(actions)}.")
		else:
			raise ValueError(f"Unknown re-formatting option {re_format}.")
	
	def get_random_actions(self, batch_size: int = 1, **kwargs) -> Any:
		as_batch = kwargs.get("as_batch", False)
		return [self.action_space.sample() for _ in range(batch_size)]
	
	def __str__(self):
		policy_repr = str(self.policy)
		tab_policy_repr = "\t" + policy_repr.replace("\n", "\n\t")
		return f"Agent<{self.behavior_name}>(\n{tab_policy_repr}\n)"
	
	def soft_update(self, policy, tau):
		self.policy.soft_update(policy, tau)
		
	def hard_update(self, policy):
		self.policy.hard_update(policy)








