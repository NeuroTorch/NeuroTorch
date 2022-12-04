"""
This tutorial is a work in progress. It will be completed in the version 0.0.1 of NeuroTorch. Stay tuned!
"""

import gym
from neurotorch.rl.agent import Agent
from neurotorch.rl.rl_academy import RLAcademy

if __name__ == '__main__':
    debug_space = gym.spaces.Dict({
       "discrete": gym.spaces.Discrete(2),
       "box": gym.spaces.Box(low=0, high=1, shape=(2,)),
       "continuous": gym.spaces.Box(low=0, high=1, shape=(10,)),
    })
    env_id = "LunarLander-v2"
    env = gym.vector.make(env_id, num_envs=4, render_mode="human")
    
    agent = Agent(
        env=env,
        behavior_name=env_id,
        policy=None,
    )
    print(agent)
    
    academy = RLAcademy(
        agent=agent,
    )
    history = academy.train(env, 10, n_epochs=3, batch_size=8, verbose=True)
    history.plot(show=True)
    
    buffer, cumulative_rewards = academy.generate_trajectories(10, epsilon=0.0, verbose=True, env=env)
    print(f"Buffer: {buffer}")
    n_terminated = sum([int(e.terminal) for e in buffer])
    print(f"{n_terminated = }")
    
    # observation, info = env.reset(seed=42)
    # for _ in range(1000):
    #    action = agent.get_actions([observation])[0]
    #    observation, reward, terminated, truncated, info = env.step(action)
    #
    #    if terminated or truncated:
    #       observation, info = env.reset()
    env.close()
