import os
from copy import deepcopy
from typing import Union, Dict

import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import neurotorch as nt

from neurotorch import Sequential, to_tensor, to_numpy
from neurotorch.modules import BaseModel
from neurotorch.rl import ReplayBuffer, PPO
from neurotorch.rl.agent import Agent as nt_Agent
from neurotorch.rl.buffers import AgentsHistoryMaps, Trajectory, Experience
from neurotorch.rl.utils import Linear, discounted_cumulative_sums
from neurotorch.transforms.base import MaybeSoftmax
from neurotorch.utils import maybe_apply_softmax


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
		# np.random.shuffle(indices)
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


class ActorNetwork(nn.Module):
	def __init__(
			self, n_actions, input_dims, alpha,
			fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'
	):
		super(ActorNetwork, self).__init__()
		
		self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
		# self.actor = Sequential(layers=[
		# 	Linear(*input_dims, fc1_dims, activation='relu'),
		# 	Linear(fc1_dims, fc2_dims, activation='relu'),
		# 	Linear(fc2_dims, n_actions, activation='softmax'),
		# ]).build()
		self.actor = nn.Sequential(
			nn.Linear(*input_dims, fc1_dims),
			nn.ReLU(),
			nn.Linear(fc1_dims, fc2_dims),
			nn.ReLU(),
			nn.Linear(fc2_dims, n_actions),
			MaybeSoftmax(dim=-1)
		)
		# self.lin0 = nn.Linear(*input_dims, fc1_dims)
		# self.lin1 = nn.Linear(fc1_dims, fc2_dims)
		# self.lin2 = nn.Linear(fc2_dims, n_actions)
		# self.activation = nn.ReLU()
		# self.softmax = nn.Softmax(dim=-1)
		
		self.optimizer = optim.Adam(self.actor.parameters(), lr=alpha)
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)
	
	def forward(self, state, **kwargs):
		dist = self.actor(state.to(self.device))
		if isinstance(dist, dict):
			dist = dist[list(dist.keys())[0]]
		# x = self.activation(self.lin0(state.to(self.device)))
		# x = self.activation(self.lin1(x))
		# dist = self.softmax(self.lin2(x))
		# dist = Categorical(dist)
		return dist
	
	def save_checkpoint(self):
		T.save(self.state_dict(), self.checkpoint_file)
	
	def load_checkpoint(self):
		self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
	def __init__(
			self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
			chkpt_dir='tmp/ppo'
	):
		super(CriticNetwork, self).__init__()
		
		self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
		# self.critic = Sequential(layers=[
		# 	Linear(*input_dims, fc1_dims, activation='relu'),
		# 	Linear(fc1_dims, fc2_dims, activation='relu'),
		# 	Linear(fc2_dims, 1, activation='identity'),
		# ]).build()
		self.critic = nn.Sequential(
			nn.Linear(*input_dims, fc1_dims),
			nn.ReLU(),
			nn.Linear(fc1_dims, fc2_dims),
			nn.ReLU(),
			nn.Linear(fc2_dims, 1)
		)
		
		self.optimizer = optim.Adam(self.critic.parameters(), lr=alpha)
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)
	
	def forward(self, state, **kwargs):
		value = self.critic(state.to(self.device))
		if isinstance(value, dict):
			value = value[list(value.keys())[0]]
		return value
	
	def save_checkpoint(self):
		T.save(self.state_dict(), self.checkpoint_file)
	
	def load_checkpoint(self):
		self.load_state_dict(T.load(self.checkpoint_file))


class Agent(nt_Agent):
	def __init__(
			self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
			policy_clip=0.2, batch_size=64, n_epochs=10, **kwargs
	):
		self.use_old = kwargs.get('use_old', False)
		self.actor = ActorNetwork(n_actions, input_dims, alpha)
		self.critic_ = CriticNetwork(input_dims, alpha)
		
		super().__init__(**kwargs)
		self.gamma = gamma
		self.policy_clip = policy_clip
		self.n_epochs = n_epochs
		self.gae_lambda = gae_lambda
		self.batch_size = batch_size
		
		self.device = self.policy.device
		param_groups = [
			{"params": self.policy.parameters(), "lr": 3e-4},
			{"params": self.critic.parameters(), "lr": 1e-3},
			{"params": self.actor.actor.parameters(), "lr": 3e-4},
			{"params": self.critic_.critic.parameters(), "lr": 1e-3}
		]
		self.optimizer = torch.optim.Adam(param_groups)
		# self.memory = PPOMemory(batch_size)
		self.memory = ReplayBuffer()
		# self.trajectory = Trajectory()
		self.agent_history_maps = AgentsHistoryMaps(self.memory)
		self.ppo = PPO(self, optimizer=self.optimizer, tau=0.0)
		self.ppo.last_agent = nt_Agent.copy_from_agent(self, requires_grad=False)
	
	def remember(self, state, action, probs, vals, reward, done):
		# self.memory.store_memory(state, action, probs, vals, reward, done)
		finished_trajectories = self.agent_history_maps.update_trajectories_(
			observations=[state],
			actions=[action],
			rewards=[reward],
			terminals=[done],
			next_observations=[state],
			others=[{'probs': probs, 'value': vals}]
		)
		self.finish_trajectories(finished_trajectories)
		# if done:
		# 	self.trajectory.append_and_terminate(
		# 		Experience(
		# 			obs=state,
		# 			reward=reward,
		# 			terminal=done,
		# 			action=action,
		# 			next_obs=state,
		# 			others={'probs': probs, 'value': vals},
		# 		)
		# 	)
		# 	self.finish_trajectories([self.trajectory])
		# 	self.memory.extend(self.trajectory)
		# 	self.trajectory = Trajectory()
		# else:
		# 	self.trajectory.append(
		# 		Experience(
		# 			obs=state,
		# 			reward=reward,
		# 			terminal=done,
		# 			action=action,
		# 			next_obs=state,
		# 			others={'probs': probs, 'value': vals},
		# 		)
		# 	)
		return
	
	def finish_trajectories(self, finished_trajectories):
		for trajectory in finished_trajectories:
			if trajectory.is_empty():
				continue
			
			if self.use_old:
				rewards = np.array([step.reward for step in trajectory])
				dones_arr = np.array([step.terminal for step in trajectory])
				values = np.array([step.others['value'] for step in trajectory])
				rewards = np.append(rewards, (1 - int(dones_arr[-1])) * values[-1])
				values = np.append(values, (1 - int(dones_arr[-1])) * values[-1])
				deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
				advantages = discounted_cumulative_sums(deltas, self.gamma * self.gae_lambda)
				adv_mean, adv_std = advantages.mean(), advantages.std()
				advantages = (advantages - adv_mean) / (adv_std + 1e-8)
				for t in range(len(trajectory)):
					trajectory.experiences[t].others['advantage'] = advantages[t]
	
				returns = discounted_cumulative_sums(rewards, self.gamma)[:-1]
				returns_mean, returns_std = returns.mean(), returns.std()
				returns = (returns - returns_mean) / (returns_std + 1e-8)
				for t in range(len(trajectory)):
					trajectory.experiences[t].others['return'] = returns[t]
			else:
				self.ppo.on_trajectory_end(self, trajectory)
	
	def save_models(self):
		print('... saving models ...')
		self.actor.save_checkpoint()
		self.critic.save_checkpoint()
	
	def load_models(self):
		print('... loading models ...')
		self.actor.load_checkpoint()
		self.critic.load_checkpoint()
	
	def choose_action(self, observation):
		state = T.tensor([observation], dtype=T.float).to(self.device)
		if self.use_old:
			dist = self.actor(state)
			dist_smax = maybe_apply_softmax(dist, dim=-1)
			dist = Categorical(dist_smax)
			action = dist.sample()
			probs = T.squeeze(dist.log_prob(action)).item()
			value = self.critic_(state)
		else:
			action, probs = self.get_actions(state, env=self.env, re_format="sample,probs", as_numpy=False)
			# dist = self.policy(state)
			value = self.critic(state)
		
		# if isinstance(dist, dict):
		# 	dist = dist[list(dist.keys())[0]]
		if isinstance(value, dict):
			value = value[list(value.keys())[0]]
		
		action = T.squeeze(action).item()
		value = T.squeeze(value).item()
		
		return action, probs, value
	
	def policy_ratio(self, batch):
		states = batch.obs
		actions = batch.actions
		
		if self.use_old:
			dist = self.actor(states)
			critic_value = self.critic_(states)
			old_probs = to_tensor(np.array([others['probs'] for others in batch.others])).to(self.device)
		else:
			dist = self.policy(states)
			# critic_value = self.critic(states)
			with torch.no_grad():
				old_dist = self.ppo.last_policy(states)
			# old_dist = self.policy(states)
			if isinstance(old_dist, dict):
				old_dist = old_dist[list(old_dist.keys())[0]]
			old_dist = maybe_apply_softmax(old_dist, dim=-1)
			# actions = torch.argmax(old_dist, dim=-1)
			old_dist = Categorical(old_dist)
			old_probs = old_dist.log_prob(actions)
		if isinstance(dist, dict):
			dist = dist[list(dist.keys())[0]]
		dist = maybe_apply_softmax(dist, dim=-1)
		dist = Categorical(dist)
		new_probs = dist.log_prob(actions)
		
		prob_ratio = torch.exp(new_probs - old_probs)
		return prob_ratio
	
	def learn(self):
		BaseModel.hard_update(self.ppo.last_policy, self.policy)
		self.finish_trajectories(self.agent_history_maps.terminate_all())
		# self.finish_trajectories([self.trajectory])
		# self.memory.extend(self.trajectory.experiences)
		# self.trajectory = Trajectory()
		for _ in range(self.n_epochs):
			for batch in self.memory.get_batch_generator(batch_size=self.batch_size, device=self.device, randomize=True):
				# advantages = self.ppo.get_advantages_from_batch(batch)
				# # prob_ratio = self.policy_ratio(batch)
				# prob_ratio = self.ppo._compute_policy_ratio(batch)
				# weighted_probs = advantages * prob_ratio
				# weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantages
				# actor_loss = -T.mean(T.min(weighted_probs, weighted_clipped_probs))
				# critic_loss = self.ppo._compute_critic_loss(batch)
				# total_loss = actor_loss + self.ppo.critic_weight * critic_loss
				# self.optimizer.zero_grad()
				# total_loss.backward()
				# self.optimizer.step()
				self.ppo.update_params(batch)

		self.memory.clear()
		# self.ppo.last_agent = nt_Agent.copy_from_agent(self, requires_grad=False)
		# BaseModel.hard_update(self.ppo.last_policy, self.policy)


def plot_learning_curve(x, scores, figure_file):
	import matplotlib.pyplot as plt
	running_avg = np.zeros(len(scores))
	for i in range(len(running_avg)):
		running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
	plt.plot(x, running_avg)
	plt.title('Running average of previous 100 scores')
	plt.show()


def main():
	import gym
	
	env = gym.make('CartPole-v1')
	N = 4096
	batch_size = 4096
	n_epochs = 80
	alpha = 0.0003
	agent = Agent(
		n_actions=env.action_space.n, batch_size=batch_size,
		alpha=alpha, n_epochs=n_epochs,
		input_dims=env.observation_space.shape,
		env=env,
	)
	n_games = 3_000
	n_itr = 30
	
	figure_file = 'plots/cartpole.png'
	
	best_score = env.reward_range[0]
	score_history = []
	
	learn_iters = 0
	avg_score = 0
	n_steps = 0
	agent.train()
	for i in range(n_games):
		observation, info = env.reset()
		terminal = False
		score = 0
		while not terminal:
			action, prob, val = agent.choose_action(observation)
			observation_, reward, done, truncated, info = env.step(action)
			terminal = done or truncated
			n_steps += 1
			score += reward
			agent.remember(observation, action, prob, val, reward, terminal)
			if n_steps % N == 0:
				agent.learn()
				learn_iters += 1
			observation = observation_
		score_history.append(score)
		avg_score = np.mean(score_history[-100:])
		
		if avg_score > best_score:
			best_score = avg_score
		# agent.save_models()
		
		print(
			'episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
			'time_steps', n_steps, 'learning_steps', learn_iters
		)
		if n_itr <= learn_iters:
			break
	x = [i + 1 for i in range(len(score_history))]
	plot_learning_curve(x, score_history, figure_file)


if __name__ == '__main__':
	main()
