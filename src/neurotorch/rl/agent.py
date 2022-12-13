from copy import deepcopy
from typing import Sequence, Union, Optional, Dict, Any, List

import numpy as np
import torch
import gym

from ..transforms.base import to_numpy, to_tensor
from ..modules.base import BaseModel
from ..modules.sequential import Sequential
from ..utils import maybe_apply_softmax

try:
	from ..modules.layers import Linear
except ImportError:
	from .utils import Linear, space_to_spec, obs_batch_to_sequence, space_to_continuous_shape, \
	get_single_observation_space, get_single_action_space, sample_action_space
from .utils import obs_sequence_to_batch


class Agent:
	@staticmethod
	def copy_from_agent(agent: "Agent", requires_grad: Optional[bool] = None) -> "Agent":
		"""
		Copy the agent.

		:param agent: The agent to copy.
		:type agent: Agent
		:param requires_grad: Whether to require gradients.
		:type requires_grad: Optional[bool]
		:return: The copied agent.
		:rtype: Agent
		"""
		return Agent(
			env=agent.env,
			observation_space=agent.observation_space,
			action_space=agent.action_space,
			behavior_name=agent.behavior_name,
			policy=agent.copy_policy(requires_grad=requires_grad),
			**agent.kwargs
		)
	
	def __init__(
			self,
			*,
			env: Optional[gym.Env] = None,
			observation_space: Optional[gym.spaces.Space] = None,
			action_space: Optional[gym.spaces.Space] = None,
			behavior_name: Optional[str] = None,
			policy: Optional[BaseModel] = None,
			policy_kwargs: Optional[Dict[str, Any]] = None,
			critic: Optional[BaseModel] = None,
			critic_kwargs: Optional[Dict[str, Any]] = None,
			**kwargs
	):
		"""
		Constructor for BaseAgent class.
		
		:param env: The environment.
		:type env: Optional[gym.Env]
		:param observation_space: The observation space. Must be a single space not batched. Must be provided if
			`env` is not provided. If `env` is provided, then this will be ignored.
		:type observation_space: Optional[gym.spaces.Space]
		:param action_space: The action space. Must be a single space not batched. Must be provided if
			`env` is not provided. If `env` is provided, then this will be ignored.
		:type action_space: Optional[gym.spaces.Space]
		:param behavior_name: The name of the behavior.
		:type behavior_name: Optional[str]
		:param policy: The model to use.
		:type policy: BaseModel
		:param policy_kwargs: The keyword arguments to pass to the policy if it is created by default.
		:type policy_kwargs: Optional[Dict[str, Any]]
		"""
		super().__init__(**kwargs)
		self.kwargs = kwargs
		self.policy_kwargs = policy_kwargs if policy_kwargs is not None else {}
		self.critic_kwargs = critic_kwargs if critic_kwargs is not None else {}
		self.env = env
		if env:
			self.observation_space = get_single_observation_space(env)
			self.action_space = get_single_action_space(env)
		else:
			self.observation_space = observation_space
			self.action_space = action_space
		if behavior_name:
			self.behavior_name = behavior_name
		elif env.spec:
			self.behavior_name = env.spec.id
		else:
			self.behavior_name = "default"
		self.policy = policy
		if self.policy is None:
			self.policy = self._create_default_policy()
		self.critic = critic
		if self.critic is None:
			self.critic = self._create_default_critic()
	
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
					output_size=self.policy_kwargs.get("default_hidden_units", 256),
					activation=self.policy_kwargs.get("default_activation", "ReLu")
				)
				for k, v in self.observation_spec.items()
			},
			*[
				Linear(
					input_size=self.policy_kwargs.get("default_hidden_units", 256),
					output_size=self.policy_kwargs.get("default_hidden_units", 256),
					activation=self.policy_kwargs.get("default_activation", "ReLu")
				)
				for _ in range(self.policy_kwargs.get("default_hidden_layers", 1))
			],
			{
				k: Linear(
					input_size=self.policy_kwargs.get("default_hidden_units", 256),
					output_size=int(space_to_continuous_shape(v, flatten_spaces=True)[0]),
					activation=self.policy_kwargs.get("default_output_activation", "Identity")
				)
				for k, v in self.action_spec.items()
			}
		],
			**self.policy_kwargs
		).build()
		return default_policy
	
	def _create_default_critic(self) -> BaseModel:
		"""
		Create the default critic.

		:return: The default critic.
		:rtype: BaseModel
		"""
		default_policy = Sequential(layers=[
			{
				k: Linear(
					input_size=int(space_to_continuous_shape(v, flatten_spaces=True)[0]),
					output_size=self.critic_kwargs.get("default_hidden_units", 256),
					activation=self.critic_kwargs.get("default_activation", "ReLu")
				)
				for k, v in self.observation_spec.items()
			},
			*[
				Linear(
					input_size=self.critic_kwargs.get("default_hidden_units", 256),
					output_size=self.critic_kwargs.get("default_hidden_units", 256),
					activation=self.critic_kwargs.get("default_activation", "ReLu")
				)
				for _ in range(self.critic_kwargs.get("default_hidden_layers", 1))
			],
			Linear(
				input_size=self.critic_kwargs.get("default_hidden_units", 256),
				output_size=self.critic_kwargs.get("default_n_values", 1),
				activation=self.critic_kwargs.get("default_output_activation", "Identity")
			)
		],
			**self.critic_kwargs
		).build()
		return default_policy

	def __call__(self, *args, **kwargs):
		"""
		Call the agent.

		:return: The output of the agent.
		"""
		return self.policy(*args, **kwargs)
	
	def get_actions(
			self,
			obs: Union[np.ndarray, torch.Tensor, Dict[str, Union[np.ndarray, torch.Tensor]]],
			**kwargs
	) -> Any:
		"""
		Get the actions for the given observations.
		
		:param obs: The observations. The observations must be batched.
		:type obs: Union[np.ndarray, torch.Tensor, Dict[str, Union[np.ndarray, torch.Tensor]]]
		:param kwargs: Keywords arguments.
		
		:keyword str re_format: The format to reformat the discrete actions to. Default is "index" which
			will return the index of the action. For other options see :mth:`format_batch_discrete_actions`.
		:keyword bool as_numpy: Whether to return the actions as numpy arrays. Default is True.
		
		:return: The actions.
		"""
		self.env = kwargs.get("env", self.env)
		re_as_dict = kwargs.get("re_as_dict", isinstance(obs, dict) or isinstance(obs[0], dict))
		re_formats = kwargs.get("re_format", "index").split(",")
		as_numpy = kwargs.get("as_numpy", True)
		
		obs_as_tensor = to_tensor(obs)
		out_actions = self.policy(obs_as_tensor, **kwargs)
		re_actions_list = [
			self.format_batch_discrete_actions(out_actions, re_format=re_format)
			for re_format in re_formats
		]
		if as_numpy:
			re_actions_list = [to_numpy(re_actions) for re_actions in re_actions_list]
		for i, re_actions in enumerate(re_actions_list):
			if not re_as_dict and isinstance(re_actions, dict):
				if not len(re_actions) == 1:
					raise ValueError("Cannot unpack actions from dict because it has not a length of 1.")
				re_actions_list[i] = re_actions[list(re_actions.keys())[0]]
			elif re_as_dict and not isinstance(re_actions, dict):
				keys = self.discrete_actions + self.continuous_actions
				re_actions_list[i] = {k: re_actions for k in keys}
		if len(re_actions_list) == 1:
			return re_actions_list[0]
		return re_actions_list
	
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
		
		TODO: fragment this method into smaller methods.
		
		:param actions: The actions.
		:param re_format: The format to reformat the actions to. Can be "logits", "probs", "index", or "one_hot".
		:param kwargs: Keywords arguments.
		:return: The formatted actions.
		"""
		discrete_actions = kwargs.get("discrete_actions", self.discrete_actions)
		actions = to_tensor(actions)
		if re_format.lower() in ["logits", "raw"]:
			return actions
		elif re_format.lower() == "probs":
			if isinstance(actions, torch.Tensor):
				return torch.softmax(actions, dim=-1) if len(discrete_actions) >= 1 else actions
			elif isinstance(actions, dict):
				return {k: (torch.softmax(v, dim=-1) if k in discrete_actions else v) for k, v in actions.items()}
			else:
				raise ValueError(f"Cannot format actions of type {type(actions)}.")
		elif re_format.lower() == "log_probs":
			if isinstance(actions, torch.Tensor):
				return torch.log_softmax(actions, dim=-1) if len(discrete_actions) >= 1 else actions
			elif isinstance(actions, dict):
				return {k: (torch.log_softmax(v, dim=-1) if k in discrete_actions else v) for k, v in actions.items()}
			else:
				raise ValueError(f"Cannot format actions of type {type(actions)}.")
		elif re_format.lower() in ["index", "indices", "argmax", "imax"]:
			if isinstance(actions, torch.Tensor):
				return torch.argmax(actions, dim=-1).long() if len(discrete_actions) >= 1 else actions
			elif isinstance(actions, dict):
				return {k: (torch.argmax(v, dim=-1).long() if k in discrete_actions else v) for k, v in actions.items()}
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
		elif re_format.lower() == "max":
			if isinstance(actions, torch.Tensor):
				return torch.max(actions, dim=-1).values if len(discrete_actions) >= 1 else actions
			elif isinstance(actions, dict):
				return {k: (torch.max(v, dim=-1).values if k in discrete_actions else v) for k, v in actions.items()}
			else:
				raise ValueError(f"Cannot format actions of type {type(actions)}.")
		elif re_format.lower() == "smax":
			if isinstance(actions, torch.Tensor):
				return torch.softmax(actions, dim=-1).max(dim=-1).values if len(discrete_actions) >= 1 else actions
			elif isinstance(actions, dict):
				return {
					k: (torch.softmax(v, dim=-1).max(dim=-1).values if k in discrete_actions else v)
					for k, v in actions.items()
				}
			else:
				raise ValueError(f"Cannot format actions of type {type(actions)}.")
		elif re_format.lower() == "log_smax":
			if isinstance(actions, torch.Tensor):
				return torch.log_softmax(actions, dim=-1).max(dim=-1).values if len(discrete_actions) >= 1 else actions
			elif isinstance(actions, dict):
				return {
					k: (torch.log_softmax(v, dim=-1).max(dim=-1).values if k in discrete_actions else v)
					for k, v in actions.items()
				}
			else:
				raise ValueError(f"Cannot format actions of type {type(actions)}.")
		elif re_format.lower() == "sample":
			if isinstance(actions, torch.Tensor):
				if len(discrete_actions) >= 1:
					probs = maybe_apply_softmax(actions, dim=-1)
					return torch.distributions.Categorical(probs=probs).sample()
				else:
					return actions
			elif isinstance(actions, dict):
				return {
					k: (
						torch.distributions.Categorical(probs=maybe_apply_softmax(v, dim=-1)).sample()
						if k in discrete_actions else v
					)
					for k, v in actions.items()
				}
			else:
				raise ValueError(f"Cannot format actions of type {type(actions)}.")
		else:
			raise ValueError(f"Unknown re-formatting option {re_format}.")
	
	def get_random_actions(self, n_samples: int = 1, **kwargs) -> Any:
		as_batch = kwargs.get("as_batch", False)
		as_sequence = kwargs.get("as_sequence", False)
		re_formats = kwargs.get("re_format", "index").split(",")
		as_numpy = kwargs.get("as_numpy", True)
		assert not (as_batch and as_sequence), "Cannot return actions as both batch and sequence."
		as_single = not (as_batch or as_sequence)
		self.env = kwargs.get("env", self.env)
		if self.env:
			action_space = self.env.action_space
		else:
			action_space = self.action_space
		if as_single and n_samples == 1:
			out_actions = sample_action_space(action_space, re_format="one_hot")
		else:
			out_actions = [sample_action_space(action_space, re_format="one_hot") for _ in range(n_samples)]
			if as_batch:
				out_actions = obs_sequence_to_batch(out_actions)
		re_actions_list = [
			self.format_batch_discrete_actions(out_actions, re_format=re_format)
			for re_format in re_formats
		]
		if as_numpy:
			re_actions_list = [to_numpy(re_actions) for re_actions in re_actions_list]
		if len(re_actions_list) == 1:
			return re_actions_list[0]
		return re_actions_list
	
	def get_values(self, obs: torch.Tensor, **kwargs) -> Any:
		"""
		Get the values for the given observations.
		
		:param obs: The batched observations.
		:param kwargs: Keywords arguments.
		:return: The values.
		"""
		self.env = kwargs.get("env", self.env)
		re_as_dict = kwargs.get("re_as_dict", isinstance(obs, dict) or isinstance(obs[0], dict))
		as_numpy = kwargs.get("as_numpy", True)
		obs_as_tensor = to_tensor(obs)
		values = self.critic(obs_as_tensor)
		if as_numpy:
			values = to_numpy(values)
		if not re_as_dict and isinstance(values, dict):
			values = values[list(values.keys())[0]]
		return values
	
	def __str__(self):
		n_tab = 2
		policy_repr = str(self.policy)
		tab_policy_repr = "\t" + policy_repr.replace("\n", "\n"+("\t"*n_tab))
		critic_repr = str(self.critic)
		tab_critic_repr = "\t" + critic_repr.replace("\n", "\n"+("\t"*n_tab))
		agent_repr = f"Agent<{self.behavior_name}>:\n\t[Policy](\n{tab_policy_repr}\t\n)\n"
		if self.critic:
			agent_repr += f"\t[Critic](\n{tab_critic_repr}\t\n)\n"
		return agent_repr
	
	def soft_update(self, policy, tau):
		self.policy.soft_update(policy, tau)
		
	def hard_update(self, policy):
		self.policy.hard_update(policy)
	
	def copy_policy(self, requires_grad: Optional[bool] = None) -> BaseModel:
		"""
		Copy the policy to a new instance.

		:return: The copied policy.
		"""
		policy_copy = deepcopy(self.policy)
		if requires_grad is not None:
			for param in policy_copy.parameters():
				param.requires_grad = requires_grad
		return policy_copy








