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
    checkpoint_manager = nt.CheckpointManager(
        checkpoint_folder=f"data/tr_data/checkpoints_{env_id}_default-policy",
        save_freq=10,
        metric=RLAcademy.CUM_REWARDS_METRIC_KEY,
        minimise_metric=False,
        save_best_only=True,
    )
    ppo_la = nt.rl.PPO(tau=0.0, critic_weight=1.0)
    
    agent = Agent(
        env=env,
        behavior_name=env_id,
        policy=None,
        policy_kwargs=dict(
            checkpoint_folder=checkpoint_manager.checkpoint_folder,
            default_hidden_units=64,
            default_activation="tanh",
        ),
        critic_kwargs=dict(
            checkpoint_folder=checkpoint_manager.checkpoint_folder,
            default_hidden_units=64,
            default_activation="tanh",
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
        n_iterations=10,
        n_epochs=80,
        n_batches=-1,
        # n_new_trajectories=1,
        n_new_experiences=4096,
        batch_size=4096,
        buffer_size=4096,
        clear_buffer=True,
        randomize_buffer=False,
        load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR,
        force_overwrite=True,
        verbose=True,
    )
    history.plot(show=True)
    # if not env.closed:
    #     env.close()
    # env = gym.vector.make(env_id, num_envs=1, render_mode="human")
    buffer, cumulative_rewards, terminal_rewards = academy.generate_trajectories(
        n_trajectories=10, epsilon=0.0, verbose=True, env=env
    )
    print(f"Buffer: {buffer}")
    n_terminated = sum([int(e.terminal) for e in buffer])
    print(f"{n_terminated = }")
    env.close()
