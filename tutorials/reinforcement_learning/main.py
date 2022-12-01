import gym
from neurotorch.rl.agent import Agent

debug_space = gym.spaces.Dict({
   "discrete": gym.spaces.Discrete(2),
   "box": gym.spaces.Box(low=0, high=1, shape=(2,)),
   "continuous": gym.spaces.Box(low=0, high=1, shape=(10,)),
})

env = gym.make("LunarLander-v2", render_mode="human")

agent = Agent(
    observation_space=env.observation_space,
    action_space=env.action_space,
    behavior_name=str(env.spec.id),
    policy=None,
)
print(agent)

observation, info = env.reset(seed=42)
for _ in range(1000):
   action = agent.get_actions([observation])[0]
   observation, reward, terminated, truncated, info = env.step(action)
   
   if terminated or truncated:
      observation, info = env.reset()
env.close()
