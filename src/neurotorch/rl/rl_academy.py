import os
import shutil
import time
import warnings
from collections import defaultdict, deque, OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union, NamedTuple

import gym
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .agent import Agent
from .buffers import ReplayBuffer, Trajectory, Experience, BatchExperience, AgentsHistoryMaps
from .ppo import PPO
from .utils import env_batch_step, env_batch_reset, batch_numpy_actions, env_batch_render
from .. import Trainer, LoadCheckpointMode, to_numpy, TrainingHistory
from ..callbacks.base_callback import BaseCallback, CallbacksList
from ..learning_algorithms.learning_algorithm import LearningAlgorithm
from ..modules import BaseModel
from ..utils import linear_decay


class GenTrajectoriesOutput(NamedTuple):
    buffer: ReplayBuffer
    cumulative_rewards: np.ndarray
    agents_history_maps: AgentsHistoryMaps
    trajectories: Optional[List[Trajectory]] = None


class RLAcademy(Trainer):
    CUM_REWARDS_METRIC_KEY = "cum_rewards"
    TERMINAL_REWARDS_METRIC_KEY = "terminal_rewards"

    def __init__(
            self,
            agent: Agent,
            *,
            predict_method: str = "__call__",
            learning_algorithm: Optional[LearningAlgorithm] = None,
            callbacks: Optional[Union[List[BaseCallback], CallbacksList, BaseCallback]] = None,
            verbose: bool = True,
            **kwargs
    ):
        self.agent = agent
        kwargs = self.set_default_academy_kwargs(**kwargs)
        super().__init__(
            model=agent,
            predict_method=predict_method,
            learning_algorithm=learning_algorithm,
            callbacks=callbacks,
            verbose=verbose,
            **kwargs
        )
        self._agents_history_maps_meta = {}

    @property
    def env(self):
        if "env" not in self.state.objects:
            raise ValueError("The environment is not set.")
        return self.state.objects["env"]

    @property
    def policy(self) -> BaseModel:
        """
        :return: The policy of the academy.
        """
        return self.model

    @staticmethod
    def set_default_academy_kwargs(**kwargs) -> Dict[str, Any]:
        """
        Set default values for the kwargs of the fit method.
        :param kwargs:
            close_env: Whether to close the environment after the training.
            n_epochs: Number of epochs to train each iteration.
            init_lr: Initial learning rate.
            min_lr: Minimum learning rate.
            weight_decay: Weight decay.
            init_epsilon: Initial epsilon. Epsilon is the probability of choosing a random action.
            epsilon_decay: Epsilon decay.
            min_epsilon: Minimum epsilon.
            gamma: Discount factor.
            tau: Target network update rate.
            n_batches: Number of batches to train each iteration.
            batch_size: Batch size.
            update_freq: Number of steps between each update.
            curriculum_strength: Strength of the teacher learning strategy.
        :return:
        """
        kwargs.setdefault("close_env", False)
        kwargs.setdefault("init_epsilon", 0.00)
        kwargs.setdefault("epsilon_decay", 0.995)
        kwargs.setdefault("min_epsilon", 0.0)
        kwargs.setdefault("n_batches", None)
        kwargs.setdefault("tau", 0.0)
        kwargs.setdefault("batch_size", 256)
        kwargs.setdefault("buffer_size", np.inf)
        kwargs.setdefault("clear_buffer", True)
        kwargs.setdefault("randomize_buffer", True)
        kwargs.setdefault("n_new_trajectories", None)
        kwargs.setdefault("n_new_experiences", kwargs["buffer_size"])
        kwargs.setdefault("clip_ratio", 0.2)
        kwargs.setdefault("use_priority_buffer", False)
        kwargs.setdefault("buffer_priority_key", "discounted_reward")
        kwargs.setdefault("normalize_rewards", False)
        kwargs.setdefault("rewards_horizon", 128)
        kwargs.setdefault("last_k_rewards", 10)
        kwargs.setdefault("last_k_rewards_key", RLAcademy.CUM_REWARDS_METRIC_KEY)

        assert kwargs["batch_size"] <= kwargs["buffer_size"]
        assert kwargs["batch_size"] > 0
        return kwargs

    def _maybe_add_learning_algorithm(self, learning_algorithm: Optional[LearningAlgorithm]) -> None:
        if len(self.learning_algorithms) == 0 and learning_algorithm is None:
            learning_algorithm = PPO(optimizer=self.optimizer, criterion=self.criterion)
        if learning_algorithm is not None:
            self.callbacks.append(learning_algorithm)

    def copy_policy(self, requires_grad: bool = False) -> BaseModel:
        """
        Copy the policy to a new instance.

        :return: The copied policy.
        """
        policy_copy = deepcopy(self.policy)
        for param in policy_copy.parameters():
            param.requires_grad = requires_grad
        policy_copy.eval()
        return policy_copy

    def copy_agent(self, requires_grad: bool = False) -> Agent:
        """
        Copy the agent to a new instance.

        :return: The copied agent.
        """
        agent_copy = self.agent.copy(requires_grad=requires_grad)
        agent_copy.policy.eval()
        return agent_copy

    def _update_optimizer_(self):
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.kwargs["init_lr"],
            weight_decay=self.kwargs["weight_decay"]
        )

    def _update_agents_history_maps_meta(self, agents_history_maps: AgentsHistoryMaps):
        self._agents_history_maps_meta = {
            "min_rewards": agents_history_maps.min_rewards,
            "max_rewards": agents_history_maps.max_rewards,
            "trajectories": agents_history_maps.trajectories,
        }

    def reset_agents_history_maps_meta(self):
        self._agents_history_maps_meta = {}
        self.update_objects_state_(observations=None, info=None)

    def generate_trajectories(
            self,
            *,
            n_trajectories: Optional[int] = None,
            n_experiences: Optional[int] = None,
            buffer: Optional[ReplayBuffer] = None,
            epsilon: float = 0.0,
            p_bar_position: int = 0,
            verbose: Optional[bool] = None,
            **kwargs
    ) -> GenTrajectoriesOutput:
        """
        Generate trajectories using the current policy. If the policy of the agent is in evaluation mode, the
        actions will be chosen with the argmax method. If the policy is in training mode and a random number is
        generated that is less than epsilon, a random action will be chosen. Otherwise, the action will be chosen
        by a sample considering the policy output.

        :param n_trajectories: Number of trajectories to generate. If not specified, the number of trajectories
            will be calculated based on the number of experiences.
        :type n_trajectories: int
        :param n_experiences: Number of experiences to generate. If not specified, the number of experiences
            will be calculated based on the number of trajectories.
        :type n_experiences: int
        :param buffer: The buffer to store the experiences.
        :type buffer: ReplayBuffer
        :param epsilon: The probability of choosing a random action.
        :type epsilon: float
        :param p_bar_position: The position of the progress bar.
        :type p_bar_position: int
        :param verbose: Whether to show the progress bar.
        :type verbose: bool
        :param kwargs: Additional arguments.

        :keyword gym.Env env: The environment to generate the trajectories. Will update the "env" of the current_state.
        :keyword observation: The initial observation. If not specified, the observation will be get from the
            the objects of the current_state attribute and if not available, the environment will be reset.
        :keyword info: The initial info. If not specified, the info will be get from the objects of the
            current_state attribute and if not available, the environment will be reset.

        :return: The buffer with the generated experiences, the cumulative rewards and the mean of terminal rewards.
        """
        if n_trajectories is None:
            n_trajectories = self.kwargs["n_new_trajectories"]
        if n_experiences is None:
            n_experiences = self.kwargs["n_new_experiences"]
        if buffer is None:
            buffer = self.current_training_state.objects.get(
                "buffer",
                ReplayBuffer(
                    self.kwargs["buffer_size"],
                    use_priority=self.kwargs["use_priority_buffer"],
                    priority_key=self.kwargs["buffer_priority_key"],
                )
            )
            self.update_objects_state_(buffer=buffer)
        if self.kwargs["clear_buffer"]:
            buffer.clear()
        if verbose is None:
            verbose = self.verbose
        if "env" in kwargs:
            if self.state.objects.get("env", None) != kwargs["env"]:
                self.reset_agents_history_maps_meta()
            self.update_objects_state_(env=kwargs["env"])
        render = kwargs.get("render", self.kwargs.get("render", False))
        rendering = [None for _ in range((self.env.num_envs if hasattr(self.env, "num_envs") else 1))]
        re_trajectories = kwargs.get("re_trajectories", False)
        re_trajectories_list = []
        agents_history_maps = AgentsHistoryMaps(
            # buffer,
            normalize_rewards=self.kwargs["normalize_rewards"], **self._agents_history_maps_meta
        )
        # terminal_rewards: List[float] = []
        p_bar = tqdm(
            total=n_experiences if n_trajectories is None else n_trajectories,
            disable=not verbose, desc="Generating Trajectories", position=p_bar_position,
            unit="trajectory" if n_trajectories is not None else "experience",
        )
        observations = kwargs.get("observations", self.current_training_state.objects.get("observations", None))
        info = kwargs.get("info", self.current_training_state.objects.get("info", None))
        if observations is None or info is None:
            observations, info = env_batch_reset(self.env)
        while not self._update_gen_trajectories_break_flag(agents_history_maps, n_trajectories, n_experiences):
            if render:
                rendering = env_batch_render(self.env)
            if not self.agent.training:
                actions = self.agent.get_actions(observations, env=self.env, re_format="argmax", as_numpy=True)
            elif np.random.random() < epsilon:
                actions = self.agent.get_random_actions(env=self.env, re_format="argmax", as_numpy=True)
            else:
                actions = self.agent.get_actions(observations, env=self.env, re_format="sample", as_numpy=True)
            actions = batch_numpy_actions(actions, self.env)
            next_observations, rewards, dones, truncated, info = env_batch_step(self.env, actions)
            terminals = np.logical_or(dones, truncated)
            finished_trajectories = agents_history_maps.update_trajectories_(
                observations=observations,
                actions=actions,
                next_observations=next_observations,
                rewards=rewards,
                terminals=terminals,
                others=[
                    {"render": rendering_item, "info": info_item, "truncated": truncated_item}
                    for rendering_item, info_item, truncated_item in zip(rendering, info, truncated)
                ],
            )
            # terminal_rewards = list(agents_history_maps.terminal_rewards.values())
            if all(terminals):
                finished_trajectories.extend(agents_history_maps.propagate_all())
                next_observations, info = env_batch_reset(self.env)
            self._update_gen_trajectories_finished_trajectories(finished_trajectories, buffer)
            if re_trajectories:
                re_trajectories_list.extend(finished_trajectories)
            if n_trajectories is None:
                p_bar.update(min(len(terminals), max(0, n_experiences - len(terminals))))
            else:
                p_bar.update(min(sum(terminals), max(0, n_trajectories - sum(terminals))))
            p_bar.set_postfix(
                cumulative_reward=f"{agents_history_maps.mean_cumulative_rewards:.3f}",
                # terminal_rewards=f"{np.nanmean(terminal_rewards) if terminal_rewards else 0.0:.3f}",
            )
            observations = next_observations
        self._update_gen_trajectories_finished_trajectories(agents_history_maps.propagate_and_get_all(), buffer)
        self._update_agents_history_maps_meta(agents_history_maps)
        self.update_objects_state_(observations=observations, info=info, buffer=buffer)
        self.update_itr_metrics_state_(
            **{
                self.CUM_REWARDS_METRIC_KEY: agents_history_maps.mean_cumulative_rewards,
                # self.TERMINAL_REWARDS_METRIC_KEY: np.mean(terminal_rewards),
            }
        )
        p_bar.close()
        return GenTrajectoriesOutput(
            buffer=buffer,
            cumulative_rewards=agents_history_maps.cumulative_rewards_as_array,
            agents_history_maps=agents_history_maps,
            trajectories=re_trajectories_list,
        )

    def _update_gen_trajectories_finished_trajectories(
            self,
            finished_trajectories: List[Trajectory],
            buffer: Optional[ReplayBuffer] = None,
    ):
        if buffer is None:
            buffer = self.state.objects.get("buffer", None)
        for trajectory in finished_trajectories:
            if not trajectory.is_empty():
                trajectory_others_list = self.callbacks.on_trajectory_end(self, trajectory)
                if trajectory_others_list is not None:
                    trajectory.update_others(trajectory_others_list)
                if buffer is not None:
                    buffer.extend(trajectory)
        return buffer

    def _update_gen_trajectories_break_flag(
            self,
            agents_history_maps: AgentsHistoryMaps,
            n_trajectories: Optional[int],
            n_experiences: Optional[int],
    ) -> bool:
        if n_trajectories is not None:
            break_flag = agents_history_maps.terminals_count >= n_trajectories
        elif n_experiences is not None:
            break_flag = agents_history_maps.experience_count >= n_experiences
        else:
            break_flag = agents_history_maps.experience_count >= agents_history_maps.buffer.capacity
        return break_flag

    def _init_train_buffer(self) -> ReplayBuffer:
        self.env.reset()
        buffer, *_ = self.generate_trajectories(n_experiences=self.kwargs["batch_size"])
        return buffer

    def train(
            self,
            env,  # TODO: add eval_env
            n_iterations: Optional[int] = None,
            *,
            n_epochs: int = 10,
            load_checkpoint_mode: LoadCheckpointMode = None,
            force_overwrite: bool = False,
            p_bar_position: Optional[int] = None,
            p_bar_leave: Optional[bool] = None,
            **kwargs
    ) -> TrainingHistory:
        self._load_checkpoint_mode = load_checkpoint_mode
        self._force_overwrite = force_overwrite
        self.kwargs.update(kwargs)
        self.update_state_(
            n_iterations=n_iterations,
            n_epochs=n_epochs,
            objects={
                **self.current_training_state.objects,
                **{"env": env, "last_k_rewards": deque(maxlen=self.kwargs["last_k_rewards"])}
            },
        )
        self.sort_callbacks_()
        self.callbacks.start(self)
        self.load_state()
        if self.current_training_state.iteration is None:
            self.update_state_(iteration=0)
        if len(self.training_history) > 0:
            self.update_itr_metrics_state_(**self.training_history.get_item_at(-1))
        else:
            self.update_state_(itr_metrics={})
        env.reset()
        # buffer = self._init_train_buffer()
        # self.update_objects_state_(buffer=buffer)
        p_bar = tqdm(
            initial=self.current_training_state.iteration,
            total=self.current_training_state.n_iterations,
            desc=kwargs.get("desc", "Training"),
            disable=not self.verbose,
            position=p_bar_position,
            unit="itr",
            leave=p_bar_leave
        )

        for i in self._iterations_generator(p_bar):
            self.update_state_(iteration=i)
            self.callbacks.on_iteration_begin(self)
            epsilon = linear_decay(
                self.kwargs["init_epsilon"], self.kwargs["min_epsilon"], self.kwargs["epsilon_decay"], i
            )
            env = self.current_training_state.objects["env"]
            env.reset()
            itr_metrics = self._exec_iteration(env, epsilon=epsilon)
            self.update_itr_metrics_state_(**itr_metrics, epsilon=epsilon)
            postfix = {f"{k}": f"{v:.5e}" for k, v in self.state.itr_metrics.items()}
            postfix.update(self.callbacks.on_pbar_update(self))
            self.callbacks.on_iteration_end(self)
            p_bar.set_postfix(postfix)
            if self.current_training_state.stop_training_flag:
                p_bar.set_postfix(OrderedDict(**{"stop_flag": "True"}, **postfix))
                break

        self.callbacks.close(self)
        p_bar.close()
        if self.kwargs.get("close_env", True):
            env.close()
        return self.training_history

    def _exec_iteration(
            self,
            env: gym.Env,
            buffer: Optional[ReplayBuffer] = None,
            **kwargs,
    ) -> Dict[str, float]:
        if buffer is None:
            buffer = self.current_training_state.objects.get("buffer", None)
        with torch.no_grad():
            torch.cuda.empty_cache()
        metrics = {}

        self.model.train()
        self.callbacks.on_train_begin(self)

        gen_trajectories_out = self.generate_trajectories(
            n_trajectories=self.kwargs["n_new_trajectories"],
            buffer=buffer,
            epsilon=kwargs.get("epsilon", 0.0),
            p_bar_position=1, verbose=False,
        )
        metrics[self.CUM_REWARDS_METRIC_KEY] = np.mean(gen_trajectories_out.cumulative_rewards)
        # metrics[self.TERMINAL_REWARDS_METRIC_KEY] = np.mean(terminal_rewards)
        self.update_state_(batch_is_train=True)
        train_losses = []
        for epoch_idx in range(self.current_training_state.n_epochs):
            self.update_state_(epoch=epoch_idx)
            train_losses.append(self._exec_epoch(gen_trajectories_out.buffer))
        train_loss = np.mean(train_losses)
        self.update_state_(train_loss=train_loss)
        self.callbacks.on_train_end(self)
        metrics["train_loss"] = train_loss

        # if val_dataloader is not None:
        # 	with torch.no_grad():
        # 		self.model.eval()
        # 		self.callbacks.on_validation_begin(self)
        # 		self.update_state_(batch_is_train=False)
        # 		val_loss = self._exec_epoch(val_dataloader)
        # 		self.update_state_(val_loss=val_loss)
        # 		self.callbacks.on_validation_end(self)
        # 		losses["val_loss"] = val_loss

        last_k_rewards = self.state.objects.get("last_k_rewards", deque(maxlen=self.kwargs["last_k_rewards"]))
        assert self.kwargs["last_k_rewards_key"] in metrics, \
            f"last_k_rewards_key {self.kwargs['last_k_rewards_key']} not in metrics. " \
            f"Please select one of {metrics.keys()}"
        last_k_rewards.append(metrics[self.kwargs["last_k_rewards_key"]])
        metrics[f"mean_last_{self.kwargs['last_k_rewards']}_rewards"] = np.nanmean(last_k_rewards)
        with torch.no_grad():
            torch.cuda.empty_cache()
        return metrics

    def _exec_epoch(
            self,
            buffer: ReplayBuffer,
    ) -> float:
        self.callbacks.on_epoch_begin(self)
        batch_size = min(len(buffer), self.kwargs["batch_size"])
        batches = buffer.get_batch_generator(
            batch_size, self.kwargs["n_batches"],
            randomize=self.kwargs["randomize_buffer"],
            device=self.agent.policy.device,
        )
        batch_losses = []
        for i, exp_batch in enumerate(batches):
            self.update_state_(batch=i)
            batch_losses.append(to_numpy(self._exec_batch(exp_batch)))
        mean_loss = np.mean(batch_losses)
        self.callbacks.on_epoch_end(self)
        return mean_loss

    def _exec_batch(
            self,
            exp_batch: BatchExperience,
            **kwargs,
    ):
        self.update_state_(x_batch=exp_batch)
        self.callbacks.on_batch_begin(self)
        if self.model.training:
            self.callbacks.on_optimization_begin(self, x=exp_batch)
            self.callbacks.on_optimization_end(self)
        else:
            self.callbacks.on_validation_batch_begin(self, x=exp_batch)
            self.callbacks.on_validation_batch_end(self)
        self.callbacks.on_batch_end(self)
        batch_loss = self.current_training_state.batch_loss
        if batch_loss is None:
            batch_loss = 0.0
        elif hasattr(batch_loss, "item") and callable(batch_loss.item):
            batch_loss = batch_loss.item()
        return batch_loss

    def close(self):
        self.env.close()
