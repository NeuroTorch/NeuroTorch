"""
This tutorial is a work in progress. It will be completed in the version 0.0.1 of NeuroTorch. Stay tuned!
"""

import gym
import numpy as np

import neurotorch as nt
from neurotorch.rl.agent import Agent
from neurotorch.rl.rl_academy import RLAcademy

if __name__ == '__main__':
    env_id = "LunarLander-v2"
    # env_id = "CartPole-v1"
    # env = gym.vector.make(env_id, num_envs=1, render_mode="human")
    env = gym.vector.make(env_id, num_envs=12, render_mode=None)
    # env = gym.make(env_id, render_mode="human")
    # env = gym.make(env_id, render_mode="rgb_array")
    checkpoint_manager = nt.CheckpointManager(
        checkpoint_folder=f"data/tr_data/checkpoints_{env_id}_default-policy",
        save_freq=10,
        metric=RLAcademy.CUM_REWARDS_METRIC_KEY,
        minimise_metric=False,
        save_best_only=True,
    )
    ppo_la = nt.rl.PPO(tau=0.0, critic_weight=0.5)
    
    agent = Agent(
        env=env,
        behavior_name=env_id,
        policy=None,
        policy_kwargs=dict(
            checkpoint_folder=checkpoint_manager.checkpoint_folder,
            default_hidden_units=256,
            default_activation="relu",
        ),
        critic_kwargs=dict(
            checkpoint_folder=checkpoint_manager.checkpoint_folder,
            default_hidden_units=256,
            default_activation="relu",
        ),
    )
    print(agent)
    
    academy = RLAcademy(
        agent=agent,
        callbacks=[checkpoint_manager, ppo_la],
        normalize_rewards=False,
        init_epsilon=0.0,
        use_priority_buffer=False,
    )
    history = academy.train(
        env,
        n_iterations=100,
        n_epochs=80,
        n_batches=-1,
        n_new_trajectories=2*env.num_envs,
        # n_new_experiences=10_000,
        batch_size=4096,
        buffer_size=np.inf,
        clear_buffer=True,
        randomize_buffer=True,
        load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR,
        force_overwrite=False,
        verbose=True,
        render=False,
    )
    history.plot(show=True)
    if not env.closed:
        env.close()
    env = gym.make(env_id, render_mode="human")
    buffer, cumulative_rewards = academy.generate_trajectories(
        n_trajectories=10, epsilon=0.0, verbose=True, env=env, render=True,
    )
    print(f"Buffer: {buffer}")
    print(f"Cumulative rewards: {np.nanmean(cumulative_rewards):.3f} +/- {np.nanstd(cumulative_rewards):.3f}")
    n_terminated = sum([int(e.terminal) for e in buffer])
    print(f"{n_terminated = }")
    env.close()
