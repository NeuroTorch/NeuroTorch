"""
This tutorial is a work in progress. It will be completed in the version 0.0.1 of NeuroTorch. Stay tuned!
"""

import gym
import numpy as np
import torch.nn

import neurotorch as nt
from neurotorch.rl.agent import Agent
from neurotorch.rl.rl_academy import RLAcademy
from neurotorch.rl.utils import TrajectoryRenderer, space_to_continuous_shape
from neurotorch.transforms.spikes_encoders import SpikesEncoder

if __name__ == '__main__':
    env_id = "LunarLander-v2"
    # env_id = "CartPole-v1"
    # env = gym.vector.make(env_id, num_envs=1, render_mode="human")
    env = gym.vector.make(env_id, num_envs=10, render_mode="rgb_array")
    # env = gym.make(env_id, render_mode="human")
    # env = gym.make(env_id, render_mode="rgb_array")
    checkpoint_manager = nt.CheckpointManager(
        checkpoint_folder=f"data/tr_data/checkpoints_{env_id}_default-policy",
        save_freq=100,
        metric=RLAcademy.CUM_REWARDS_METRIC_KEY,
        minimise_metric=False,
        save_best_only=True,
    )
    ppo_la = nt.rl.PPO(
        tau=0.0,
        critic_weight=0.5,
        gae_lambda=0.99,
        default_critic_lr=5e-4,
        default_policy_lr=5e-4,
        critic_criterion=torch.nn.SmoothL1Loss(),
    )
    
    agent = Agent(
        env=env,
        behavior_name=env_id,
        # policy=nt.SequentialRNN(
        #     input_transform=[
        #         SpikesEncoder(
        #             n_steps=32,
        #             n_units=space_to_continuous_shape(env.single_observation_space)[0],
        #             spikes_layer_type=nt.SpyLIFLayer,
        #         )
        #     ],
        #     layers=[
        #         nt.SpyLIFLayer(
        #             space_to_continuous_shape(env.single_observation_space)[0], 128, use_recurrent_connection=False
        #         ),
        #         nt.SpyLILayer(128, space_to_continuous_shape(env.single_action_space)[0]),
        #     ],
        #     output_transform=[nt.transforms.ReduceMax(dim=1)],
        # ).build(),
        policy=nt.Sequential(
            layers=[
                torch.nn.Linear(space_to_continuous_shape(env.single_observation_space)[0], 128),
                torch.nn.Dropout(0.1),
                torch.nn.PReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.Dropout(0.1),
                torch.nn.PReLU(),
                torch.nn.Linear(128, space_to_continuous_shape(env.single_action_space)[0]),
            ]
        ),
        policy_kwargs=dict(
            checkpoint_folder=checkpoint_manager.checkpoint_folder,
            default_hidden_units=[128, 128],
            default_activation=torch.nn.PReLU(),
        ),
        critic=nt.Sequential(
            layers=[
                torch.nn.Linear(space_to_continuous_shape(env.single_observation_space)[0], 128),
                torch.nn.Dropout(0.1),
                torch.nn.PReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.Dropout(0.1),
                torch.nn.PReLU(),
                torch.nn.Linear(128, 1),
            ]
        ),
        critic_kwargs=dict(
            checkpoint_folder=checkpoint_manager.checkpoint_folder,
            default_hidden_units=[128, 128],
            default_activation=torch.nn.PReLU(),
        ),
    )
    print(agent)
    
    academy = RLAcademy(
        agent=agent,
        callbacks=[checkpoint_manager, ppo_la],
        normalize_rewards=False,
        init_epsilon=0.00,
        use_priority_buffer=False,
    )
    history = academy.train(
        env,
        n_iterations=100,
        n_epochs=50,
        n_batches=-1,
        n_new_trajectories=env.num_envs,
        # n_new_experiences=10_000,
        batch_size=-1,
        buffer_size=np.inf,
        clear_buffer=True,
        randomize_buffer=False,
        load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR,
        force_overwrite=True,
        verbose=True,
        render=False,
        last_k_rewards=10,
    )
    history.plot(show=True)
    if not env.closed:
        env.close()
    agent.load_checkpoint(
        checkpoints_meta_path=checkpoint_manager.checkpoints_meta_path,
        load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR
    )
    env = gym.make(env_id, render_mode="rgb_array")
    gen_trajectories_out = academy.generate_trajectories(
        n_trajectories=1, epsilon=0.0, verbose=True, env=env, render=True, re_trajectories=True,
    )
    for i, trajectory in enumerate(gen_trajectories_out.trajectories):
        trajectory_renderer = TrajectoryRenderer(trajectory=trajectory, env=env)
        trajectory_renderer.render()
        trajectory_renderer.to_mp4(f"figures/trajectory_{i}.mp4")
    cumulative_rewards = gen_trajectories_out.cumulative_rewards
    print(f"Buffer: {gen_trajectories_out.buffer}")
    print(f"Cumulative rewards: {np.nanmean(cumulative_rewards):.3f} +/- {np.nanstd(cumulative_rewards):.3f}")
    n_terminated = sum([int(e.terminal) for e in gen_trajectories_out.buffer])
    print(f"{n_terminated = }")
    env.close()
