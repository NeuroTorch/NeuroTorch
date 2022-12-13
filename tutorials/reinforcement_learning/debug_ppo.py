"""
This tutorial is a work in progress. It will be completed in the version 0.0.1 of NeuroTorch. Stay tuned!
"""
from typing import List

import gym
import numpy as np
import torch
from tqdm import tqdm

from neurotorch.rl.agent import Agent
from neurotorch.rl.buffers import AgentsHistoryMaps, ReplayBuffer
from neurotorch.rl.utils import env_batch_step


class PPOMemory:
	def __init__(self, batch_size):
		self.states = []
		self.probs = []
		self.vals = []
		self.actions = []
		self.rewards = []
		self.dones = []
		
		self.batch_size = batch_size
	
	def generate_batches(self):
		n_states = len(self.states)
		batch_start = np.arange(0, n_states, self.batch_size)
		indices = np.arange(n_states, dtype=np.int64)
		np.random.shuffle(indices)
		batches = [indices[i:i + self.batch_size] for i in batch_start]
		
		return np.array(self.states), \
			np.array(self.actions), \
			np.array(self.probs), \
			np.array(self.vals), \
			np.array(self.rewards), \
			np.array(self.dones), \
			batches
	
	def store_memory(self, state, action, probs, vals, reward, done):
		self.states.append(state)
		self.actions.append(action)
		self.probs.append(probs)
		self.vals.append(vals)
		self.rewards.append(reward)
		self.dones.append(done)
	
	def clear_memory(self):
		self.states = []
		self.probs = []
		self.actions = []
		self.rewards = []
		self.dones = []
		self.vals = []


class PPOAgent(Agent):
	def __init__(self):
		super().__init__()
		self.memory = PPOMemory(batch_size=64)
		self.gamma = 0.99
		self.gae_lambda = 0.95
		self.eps_clip = 0.2
		self.epochs = 10
		self.batches = 5
		self.critic_weight = 0.5
		self.tau = 0.95
	
	@property
	def actor(self):
		return self.policy
	
	def learn(self, n_epochs=30):
		for _ in range(n_epochs):
			state_arr, action_arr, old_prob_arr, vals_arr, \
				reward_arr, dones_arr, batches = \
				self.memory.generate_batches()
			
			values = vals_arr
			advantage = np.zeros(len(reward_arr), dtype=np.float32)
			
			for t in range(len(reward_arr) - 1):
				discount = 1
				a_t = 0
				for k in range(t, len(reward_arr) - 1):
					a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
					discount *= self.gamma * self.gae_lambda
				advantage[t] = a_t
			advantage = torch.tensor(advantage).to(self.actor.device)
			
			values = torch.tensor(values).to(self.actor.device)
			for batch in batches:
				states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
				old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
				actions = torch.tensor(action_arr[batch]).to(self.actor.device)
				
				dist = self.actor(states)
				critic_value = self.critic(states)
				
				critic_value = torch.squeeze(critic_value)
				
				new_probs = dist.log_prob(actions)
				prob_ratio = new_probs.exp() / old_probs.exp()
				weighted_probs = advantage[batch] * prob_ratio
				weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage[batch]
				actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
				
				returns = advantage[batch] + values[batch]
				critic_loss = (returns - critic_value)**2
				critic_loss = critic_loss.mean()
				
				total_loss = actor_loss + 0.5 * critic_loss
				self.actor.optimizer.zero_grad()
				self.critic.optimizer.zero_grad()
				total_loss.backward()
				self.actor.optimizer.step()
				self.critic.optimizer.step()
		
		self.memory.clear_memory()


if __name__ == '__main__':
	# env_id = "LunarLander-v2"
	env_id = "CartPole-v1"
	# env = gym.vector.make(env_id, num_envs=1, render_mode="human")
	env = gym.vector.make(env_id, num_envs=1, render_mode=None)
	# env = gym.make(env_id, render_mode="human")
	
	agent = Agent(
		env=env,
		behavior_name=env_id,
		policy=None,
		policy_kwargs=dict(
			# checkpoint_folder=checkpoint_manager.checkpoint_folder,
			default_hidden_units=64,
			default_activation="tanh",
		),
		critic_kwargs=dict(
			# checkpoint_folder=checkpoint_manager.checkpoint_folder,
			default_hidden_units=64,
			default_activation="tanh",
		),
	)
	memory = PPOMemory(batch_size=5)
	print(agent)
	n_trajectories = 300
	buffer = ReplayBuffer(20, use_priority=False)
	agents_history_maps = AgentsHistoryMaps(buffer, normalize_rewards=False)
	cumulative_rewards: List[float] = []
	terminal_rewards: List[float] = []
	p_bar = tqdm(
		total=n_trajectories,
		disable=False, desc="Generating Trajectories", position=0,
		unit="trajectory" if n_trajectories is not None else "experience"
	)
	
	observations, info = env.reset()
	for i in range(n_trajectories):
		actions_index, actions_probs = agent.get_actions(observations, env=env, re_format="index,probs")
		values = agent.get_values(observations, env=env)
		next_observations, rewards, dones, truncated, infos = env_batch_step(env, actions_index)
		finished_trajectories = agents_history_maps.update_trajectories_(
			observations=observations,
			actions=actions_probs,
			next_observations=next_observations,
			rewards=rewards,
			dones=dones,
		)
		cumulative_rewards = list(agents_history_maps.cumulative_rewards.values())
		terminal_rewards = list(agents_history_maps.terminal_rewards.values())
		# self._update_gen_trajectories_finished_trajectories(finished_trajectories)
		if all(dones):
			agents_history_maps.terminate_all()
			next_observations, info = env.reset()
		p_bar.update(min(sum(dones), max(0, n_trajectories - sum(dones))))
		p_bar.set_postfix(
			cumulative_reward=f"{np.mean(cumulative_rewards) if cumulative_rewards else 0.0:.3f}",
			terminal_rewards=f"{np.mean(terminal_rewards) if terminal_rewards else 0.0:.3f}",
		)
		observations = next_observations
	# self._update_gen_trajectories_finished_trajectories(agents_history_maps.terminate_all())
	# self._update_agents_history_maps_meta(agents_history_maps)
	# self.update_objects_state_(observations=observations, info=info, buffer=buffer)
	# self.update_itr_metrics_state_(
	# 	**{
	# 		self.CUM_REWARDS_METRIC_KEY     : np.mean(cumulative_rewards),
	# 		self.TERMINAL_REWARDS_METRIC_KEY: np.mean(terminal_rewards),
	# 	}
	# )
	p_bar.close()
