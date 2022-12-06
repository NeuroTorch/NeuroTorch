"""
This tutorial is a work in progress. It will be completed in the version 0.0.1 of NeuroTorch. Stay tuned!
"""

import gym
import neurotorch as nt
from neurotorch.rl.agent import Agent
from neurotorch.rl.rl_academy import RLAcademy

if __name__ == '__main__':
    env_id = "LunarLander-v2"
    env = gym.vector.make(env_id, num_envs=6, render_mode="human")
    checkpoint_manager = nt.CheckpointManager(
        checkpoint_folder=f"data/tr_data/checkpoints_{env_id}_default-policy",
        save_freq=10,
        metric=RLAcademy.REWARD_METRIC_KEY,
        minimise_metric=False,
        save_best_only=True,
    )
    
    agent = Agent(
        env=env,
        behavior_name=env_id,
        policy=None,
        policy_kwargs=dict(checkpoint_folder=checkpoint_manager.checkpoint_folder),
    )
    print(agent)
    
    academy = RLAcademy(
        agent=agent,
        callbacks=[checkpoint_manager],
        normalize_rewards=True,
    )
    history = academy.train(
        env,
        n_iterations=100,
        n_epochs=3,
        batch_size=256,
        load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR,
        force_overwrite=True,
        verbose=True,
    )
    history.plot(show=True)
    
    buffer, cumulative_rewards = academy.generate_trajectories(10, epsilon=0.0, verbose=True, env=env)
    print(f"Buffer: {buffer}")
    n_terminated = sum([int(e.terminal) for e in buffer])
    print(f"{n_terminated = }")
    env.close()
