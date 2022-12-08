"""
This tutorial is a work in progress. It will be completed in the version 0.0.1 of NeuroTorch. Stay tuned!
"""
from typing import List

import gym
import numpy as np
from tqdm import tqdm

import neurotorch as nt
from neurotorch.rl.agent import Agent
from neurotorch.rl.buffers import AgentsHistoryMaps, ReplayBuffer
from neurotorch.rl.rl_academy import RLAcademy
from neurotorch.rl.utils import env_batch_step

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

