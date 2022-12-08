"""
This tutorial is a work in progress. It will be completed in the version 0.0.1 of NeuroTorch. Stay tuned!
"""

import gym
import neurotorch as nt
from neurotorch.rl.agent import Agent
from neurotorch.rl.rl_academy import RLAcademy

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
	
	

