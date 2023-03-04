"""
This tutorial is a work in progress. It will be completed in the version 0.0.1 of NeuroTorch. Stay tuned!
"""
import os

import gym
import numpy as np
import torch.nn

import neurotorch as nt
from neurotorch import CheckpointManager
from neurotorch.callbacks.early_stopping import EarlyStoppingThreshold
from neurotorch.rl import PPO
from neurotorch.rl.agent import Agent
from neurotorch.rl.rl_academy import RLAcademy
from neurotorch.rl.utils import TrajectoryRenderer, space_to_continuous_shape
from neurotorch.transforms.spikes_encoders import SpikesEncoder


def get_env_config_desc(env_config: dict):
    desc = f"{env_config['id']}"
    if env_config["continuous"]:
        desc += f"_continuous"
    else:
        desc += f"_discrete"
    return desc


def set_default_agent_config(agent_config: dict):
    if agent_config is None:
        agent_config = {}
    agent_config.setdefault("use_spiking_policy", False)
    agent_config.setdefault("n_hidden_units", 128)
    agent_config.setdefault("n_critic_hidden_units", 128)
    agent_config.setdefault("n_encoder_steps", 8)
    return agent_config


def set_default_trainer_config(trainer_config: dict):
    if trainer_config is None:
        trainer_config = {}
    trainer_config.setdefault("n_iterations", 5_000)
    trainer_config.setdefault("n_epochs", 30)
    trainer_config.setdefault("n_new_trajectories", 5)
    trainer_config.setdefault("last_k_rewards", 10)
    return trainer_config


def get_agent_model(env, env_config, agent_config: dict):
    agent_config = set_default_agent_config(agent_config)
    use_spiking_policy = agent_config["use_spiking_policy"]
    n_hidden_units = agent_config["n_hidden_units"]
    n_critic_hidden_units = agent_config["n_critic_hidden_units"]
    n_encoder_steps = agent_config["n_encoder_steps"]
    continuous_obs_shape = space_to_continuous_shape(getattr(env, "single_observation_space", env.observation_space))
    continuous_action_shape = space_to_continuous_shape(getattr(env, "single_action_space", env.action_space))
    
    hash_id = nt.utils.hash_params(agent_config)
    env_desc = get_env_config_desc(env_config)
    if use_spiking_policy:
        checkpoint_folder = f"data/tr_data/chkp_{env_desc}-snn_{hash_id}"
    else:
        checkpoint_folder = f"data/tr_data/chkp_{env_desc}-{hash_id}"
    
    if use_spiking_policy:
        policy = nt.SequentialRNN(
            input_transform=[
                # SpikesEncoder(
                #     n_steps=n_encoder_steps,
                #     n_units=continuous_obs_shape[0],
                #     spikes_layer_type=nt.SpyLIFLayer,
                # )
                nt.transforms.ConstantValuesTransform(n_steps=n_encoder_steps)
            ],
            layers=[
                nt.SpyLIFLayerLPF(
                    continuous_obs_shape[0], n_hidden_units, use_recurrent_connection=False
                ),
                nt.SpyLILayer(n_hidden_units, continuous_action_shape[0]),
            ],
            output_transform=[
                (
                    nt.transforms.ReduceFuncTanh(nt.transforms.ReduceMean(dim=1))
                    if env_config["continuous"] else
                    nt.transforms.ReduceMax(dim=1)
                )
            ],
        ).build()
    else:
        policy = nt.Sequential(
            layers=[
                torch.nn.Linear(continuous_obs_shape[0], n_hidden_units),
                torch.nn.Dropout(0.1),
                torch.nn.PReLU(),
                torch.nn.Linear(n_hidden_units, n_hidden_units),
                torch.nn.Dropout(0.1),
                torch.nn.PReLU(),
                torch.nn.Linear(n_hidden_units, continuous_action_shape[0]),
                (torch.nn.Tanh() if env_config["continuous"] else torch.nn.Identity())
            ]
        ).build()
    
    agent = Agent(
        env=env,
        behavior_name=env_config["id"],
        policy=policy,
        critic=nt.Sequential(
            layers=[
                torch.nn.Linear(continuous_obs_shape[0], n_critic_hidden_units),
                torch.nn.Dropout(0.1),
                torch.nn.PReLU(),
                torch.nn.Linear(n_critic_hidden_units, n_critic_hidden_units),
                torch.nn.Dropout(0.1),
                torch.nn.PReLU(),
                torch.nn.Linear(n_critic_hidden_units, 1),
            ]
        ).build(),
        checkpoint_folder=checkpoint_folder,
    )
    return agent


def train(env_config, *, agent_config: dict = None, trainer_config: dict = None, **kwargs):
    trainer_config = set_default_trainer_config(trainer_config)
    n_iterations = trainer_config["n_iterations"]
    n_epochs = trainer_config["n_epochs"]
    n_new_trajectories = trainer_config["n_new_trajectories"]
    last_k_rewards = trainer_config["last_k_rewards"]
    env_config["render_mode"] = "rgb_array"
    
    force_overwrite = kwargs.get("force_overwrite", False)
    use_multiprocessing = kwargs.get("use_multiprocessing", True)
    
    if use_multiprocessing:
        env = gym.vector.make(num_envs=n_new_trajectories, **env_config)
    else:
        env = gym.make(**env_config)
    
    agent = get_agent_model(env, env_config, agent_config)
    checkpoint_manager = CheckpointManager(
        checkpoint_folder=agent.checkpoint_folder,
        checkpoints_meta_path=agent.checkpoints_meta_path,
        save_freq=int(0.1 * n_iterations),
        metric=RLAcademy.CUM_REWARDS_METRIC_KEY,
        minimise_metric=False,
        save_best_only=False,
    )
    early_stopping = EarlyStoppingThreshold(
        metric=f"mean_last_{last_k_rewards}_rewards",
        threshold=230.0,
        minimize_metric=False,
    )
    ppo_la = PPO(
        critic_criterion=torch.nn.SmoothL1Loss(),
    )
    academy = RLAcademy(
        agent=agent,
        callbacks=[checkpoint_manager, ppo_la, early_stopping],
    )
    print(f"Academy:\n{academy}")
    os.makedirs(f"{agent.checkpoint_folder}/infos", exist_ok=True)
    with open(f"{agent.checkpoint_folder}/infos/academy.txt", "w+") as f:
        f.write(repr(academy))
    
    history = academy.train(
        env,
        n_iterations=n_iterations,
        n_epochs=n_epochs,
        n_batches=-1,
        n_new_trajectories=n_new_trajectories,
        batch_size=4096,
        buffer_size=np.inf,
        clear_buffer=True,
        randomize_buffer=True,
        load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR,
        force_overwrite=force_overwrite,
        verbose=True,
        render=False,
        last_k_rewards=last_k_rewards,
    )
    if not getattr(env, "closed", False):
        env.close()
    
    history.plot(show=True)
    
    try:
        agent.load_checkpoint(load_checkpoint_mode=nt.LoadCheckpointMode.BEST_ITR)
    except:
        agent.load_checkpoint(load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR)
    env = gym.make(**env_config)
    agent.eval()
    gen_trajectories_out = academy.generate_trajectories(
        n_trajectories=10, epsilon=0.0, verbose=True, env=env, render=True, re_trajectories=True,
    )
    best_trajectory_idx = np.argmax([t.cumulative_reward for t in gen_trajectories_out.trajectories])
    trajectory_renderer = TrajectoryRenderer(trajectory=gen_trajectories_out.trajectories[best_trajectory_idx], env=env)
    
    cumulative_rewards = gen_trajectories_out.cumulative_rewards
    print(f"Buffer: {gen_trajectories_out.buffer}")
    print(f"Cumulative rewards: {np.nanmean(cumulative_rewards):.3f} +/- {np.nanstd(cumulative_rewards):.3f}")
    best_cum_reward_fmt = f"{cumulative_rewards[best_trajectory_idx]:.3f}"
    print(f"Best trajectory: {best_trajectory_idx}, cumulative reward: {best_cum_reward_fmt}")
    trajectory_renderer.render()
    
    trajectory_renderer.to_mp4(
        f"{agent.checkpoint_folder}/figures/trajectory_{best_trajectory_idx}-"
        f"CR{best_cum_reward_fmt.replace('.', '_')}.mp4"
    )


if __name__ == '__main__':
    train(
        env_config={
            "id": "LunarLander-v2",
            "continuous": True,
            "render_mode": "rgb_array",
            "gravity": -10.0,
            "enable_wind": True,
            "wind_power": 5.0,
            "turbulence_power": 0.5
        },
        trainer_config={
            "n_iterations"      : 1_000,
            "n_epochs"          : 30,
            "n_new_trajectories": 5,
        },
        agent_config={
          "use_spiking_policy": True,
        },
        force_overwrite=True,
        use_multiprocessing=True,
    )
