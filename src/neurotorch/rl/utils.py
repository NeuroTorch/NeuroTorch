import os
import warnings
from collections import defaultdict
from typing import Optional, Union, Sequence, Dict, Tuple, Any, List

import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation
import numpy as np
import gym
import scipy
import torch

from .buffers import Trajectory
from .curriculum import Curriculum
from ..dimension import SizeTypes
from ..modules.layers import BaseNeuronsLayer
from ..transforms.base import to_tensor, to_numpy
from ..callbacks import TrainingHistory
from ..utils import legend_without_duplicate_labels_


class TrainingHistoriesMap:
    REPORT_KEY = "report"

    def __init__(self, curriculum: Optional[Curriculum] = None):
        self.curriculum = curriculum
        self.histories = defaultdict(TrainingHistory, **{TrainingHistoriesMap.REPORT_KEY: TrainingHistory()})

    @property
    def report_history(self) -> TrainingHistory:
        return self.histories[TrainingHistoriesMap.REPORT_KEY]

    def max(self, key=None):
        if self.curriculum is None:
            return self.histories[TrainingHistoriesMap.REPORT_KEY].max(key)
        else:
            return self.histories[self.curriculum.current_lesson.name].max(key)

    def concat(self, other):
        self.histories[TrainingHistoriesMap.REPORT_KEY].concat(other)
        if self.curriculum is not None:
            return self.histories[self.curriculum.current_lesson.name].concat(other)

    def append(self, key, value):
        self.histories[TrainingHistoriesMap.REPORT_KEY].append(key, value)
        if self.curriculum is not None:
            return self.histories[self.curriculum.current_lesson.name].append(key, value)

    @staticmethod
    def _set_default_plot_kwargs(kwargs: dict):
        kwargs.setdefault('fontsize', 16)
        kwargs.setdefault('linewidth', 3)
        kwargs.setdefault('figsize', (16, 12))
        kwargs.setdefault('dpi', 300)
        return kwargs

    def plot(self, save_path=None, show=False, lesson_idx: Optional[Union[int, str]] = None, **kwargs):
        kwargs = self._set_default_plot_kwargs(kwargs)
        if self.curriculum is None:
            assert lesson_idx is None, "lesson_idx must be None if curriculum is None"
            return self.plot_history(TrainingHistoriesMap.REPORT_KEY, save_path, show, **kwargs)
        if lesson_idx is None:
            self.plot_history(TrainingHistoriesMap.REPORT_KEY, save_path, show, **kwargs)
        else:
            self.plot_history(self.curriculum[lesson_idx].name, save_path, show, **kwargs)

    def plot_history(
            self,
            history_name: str,
            save_path=None,
            show=False,
            **kwargs
    ):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        history = self.histories[history_name]
        if self.curriculum is not None and history_name != TrainingHistoriesMap.REPORT_KEY:
            lessons = [self.curriculum[history_name]]
            lessons_start_itr = [0]
        elif self.curriculum is not None and history_name == TrainingHistoriesMap.REPORT_KEY:
            lessons = self.curriculum.lessons
            lessons_lengths = {k: [len(self.histories[lesson.name][k]) for lesson in lessons] for k in history._container}
            lessons_start_itr = {k: np.cumsum(lessons_lengths[k]) for k in history.keys()}
        else:
            lessons = []
            lessons_start_itr = []

        kwargs = self._set_default_plot_kwargs(kwargs)
        loss_metrics = [k for k in history.keys() if 'loss' in k.lower()]
        rewards_metrics = [k for k in history.keys() if 'reward' in k.lower()]
        other_metrics = [k for k in history.keys() if k not in loss_metrics and k not in rewards_metrics]
        n_metrics = 2 + len(other_metrics)
        n_cols = int(np.sqrt(n_metrics))
        n_rows = int(n_metrics / n_cols)
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=kwargs["figsize"], sharex='all')
        if axes.ndim == 1:
            axes = np.expand_dims(axes, axis=-1)
        for row_i in range(n_rows):
            for col_i in range(n_cols):
                ax = axes[row_i, col_i]
                ravel_index = row_i * n_cols + col_i
                if ravel_index == 0:
                    for k in loss_metrics:
                        ax.plot(history[k], label=k, linewidth=kwargs['linewidth'])
                    ax.set_ylabel("Loss [-]", fontsize=kwargs["fontsize"])
                    ax.legend(fontsize=kwargs["fontsize"])
                elif ravel_index == 1:
                    for k in rewards_metrics:
                        ax.plot(history[k], label=k, linewidth=kwargs['linewidth'])
                        for lesson_idx, lesson in enumerate(lessons):
                            if lesson.completion_criteria.measure == k:
                                ax.plot(
                                    lesson.completion_criteria.threshold*np.ones(len(history[k])), 'k--',
                                    label=f"{k} threshold", linewidth=kwargs['linewidth']
                                )
                            if history_name == TrainingHistoriesMap.REPORT_KEY and lesson.is_completed:
                                ax.axvline(
                                    lessons_start_itr[k][lesson_idx], ymin=np.min(history[k]), ymax=np.max(history[k]),
                                    color='r', linestyle='--', linewidth=kwargs['linewidth'], label=f"lesson start"
                                )
                    ax.set_ylabel("Rewards [-]", fontsize=kwargs["fontsize"])
                    ax.legend(fontsize=kwargs["fontsize"])
                else:
                    k = other_metrics[ravel_index - 1]
                    ax.plot(history[k], label=k, linewidth=kwargs['linewidth'])
                    ax.legend(fontsize=kwargs["fontsize"])
                if row_i == n_rows - 1:
                    ax.set_xlabel("Iterations [-]", fontsize=kwargs["fontsize"])
                legend_without_duplicate_labels_(ax)
        if save_path is not None:
            plt.savefig(save_path, dpi=kwargs["dpi"])
        if show:
            plt.show()
        plt.close(fig)


def space_to_spec(space: gym.spaces.Space) -> Dict[str, gym.spaces.Space]:
    spec = {}
    if hasattr(space, 'spaces'):
        for k, v in space.spaces.items():
            spec[k] = space_to_spec(v)
    else:
        spec[str(type(space).__name__)] = space
    return spec


def space_to_continuous_shape(
        space: gym.spaces.Space,
        flatten_spaces=False
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    if hasattr(space, 'spaces'):
        shapes = {}
        for k, v in space.spaces.items():
            shapes[k] = space_to_continuous_shape(v)
        return shapes
    else:
        if isinstance(space, (gym.spaces.Discrete, )):
            return (space.n, )
        else:
            shape = space.shape
            if flatten_spaces:
                shape = (np.prod(shape), )
            return shape


def obs_sequence_to_batch(
        obs: Sequence[Union[np.ndarray, torch.Tensor, Dict[str, Union[np.ndarray, torch.Tensor]]]]
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Convert a sequence of observations to a batch of observations.

    :param obs: The sequence of observations.
    :type obs: Sequence[Union[np.ndarray, torch.Tensor, Dict[str, Union[np.ndarray, torch.Tensor]]]]

    :return: The batch of observations.
    :rtype: Union[torch.Tensor, Dict[str, torch.Tensor]]
    """
    obs_as_tensor = [to_tensor(obs_i) for obs_i in obs]
    if isinstance(obs_as_tensor[0], dict):
        return {k: torch.stack([o[k] for o in obs_as_tensor]) for k in obs[0].keys()}
    else:
        return torch.stack(obs_as_tensor)


def obs_batch_to_sequence(
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        as_numpy: bool = False
) -> Sequence[Union[np.ndarray, torch.Tensor, Dict[str, Union[np.ndarray, torch.Tensor]]]]:
    """
    Convert a batch of observations to a sequence of observations.

    :param obs: The batch of observations.
    :type obs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    :param as_numpy: Whether to convert the observations to numpy arrays.
    :type as_numpy: bool

    :return: The sequence of observations.
    :rtype: Sequence[Union[np.ndarray, torch.Tensor, Dict[str, Union[np.ndarray, torch.Tensor]]]]
    """
    if as_numpy:
        obs = to_numpy(obs)
    if isinstance(obs, dict):
        return [{k: obs[k][i] for k in obs.keys()} for i in range(obs[list(obs.keys())[0]].shape[0])]
    else:
        return [obs[i] for i in range(obs.shape[0])]


class Linear(BaseNeuronsLayer):
    def __init__(
            self,
            input_size: Optional[SizeTypes] = None,
            output_size: Optional[SizeTypes] = None,
            name: Optional[str] = None,
            device: Optional[torch.device] = None,
            **kwargs
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            name=name,
            use_recurrent_connection=False,
            device=device,
            **kwargs
        )
        self.bias_weights = None
        self.activation = self._init_activation(self.kwargs["activation"])

    def _set_default_kwargs(self):
        self.kwargs.setdefault("use_bias", True)
        self.kwargs.setdefault("activation", "identity")

    def _init_activation(self, activation: Union[torch.nn.Module, str]):
        """
        Initialise the activation function.
        :param activation: Activation function.
        :type activation: Union[torch.nn.Module, str]
        """
        str_to_activation = {
            "identity": torch.nn.Identity(),
            "relu"    : torch.nn.ReLU(),
            "tanh"    : torch.nn.Tanh(),
            "sigmoid" : torch.nn.Sigmoid(),
            "softmax" : torch.nn.Softmax(dim=-1),
        }
        if isinstance(activation, str):
            activation = activation.lower()
            assert activation in str_to_activation.keys(), f"Activation {activation} is not implemented."
            self.activation = str_to_activation[activation]
        else:
            self.activation = activation
        return self.activation

    def build(self) -> 'Linear':
        if self.kwargs["use_bias"]:
            self.bias_weights = torch.nn.Parameter(
                torch.empty((int(self.output_size),), device=self.device),
                requires_grad=self.requires_grad,
            )
        else:
            self.bias_weights = torch.zeros((int(self.output_size),), dtype=torch.float32, device=self.device)
        super().build()
        self.initialize_weights_()
        return self

    def initialize_weights_(self):
        super().initialize_weights_()
        torch.nn.init.kaiming_uniform_(self._forward_weights.data, a=np.sqrt(5))
        if "bias_weights" in self.kwargs:
            self.bias_weights.data = to_tensor(self.kwargs["bias_weights"]).to(self.device)
        else:
            # torch.nn.init.constant_(self.bias_weights, 0.0)
            bound = 1 / np.sqrt(int(self.input_size))
            torch.nn.init.uniform_(self.bias_weights, -bound, bound)

    def create_empty_state(
            self,
            batch_size: int = 1,
            **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        kwargs.setdefault("n_hh", 0)
        return super().create_empty_state(batch_size=batch_size, **kwargs)

    def forward(
            self,
            inputs: torch.Tensor,
            state: Tuple[torch.Tensor, ...] = None,
            **kwargs
    ):
        # assert inputs.ndim == 2
        # batch_size, nb_features = inputs.shape
        # x = torch.functional.F.linear(inputs, self.forward_weights.T, self.bias_weights)
        x = torch.matmul(inputs, self.forward_weights) + self.bias_weights
        return self.activation(x)


def env_batch_step(
        env: gym.Env,
        actions: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Step the environment in batch mode.

    :param env: The environment.
    :type env: gym.Env
    :param actions: The actions to take.
    :type actions: Any

    :return: The batch of observations, rewards, dones, truncated and infos.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    # actions_as_numpy = to_numpy(actions).reshape(-1).tolist()  # TODO: to numpy without changing the dtype
    if isinstance(actions, dict) and len(actions) == 1:
        actions = list(actions.values())[0]
    actions_as_numpy = actions
    if isinstance(env, gym.vector.VectorEnv):
        observations, rewards, dones, truncateds, info = env.step(actions_as_numpy)
        infos = [info for _ in range(env.num_envs)]
    else:
        actions_as_single = actions_as_numpy[0] if actions_as_numpy.ndim > 0 else actions_as_numpy
        observation, reward, done, truncated, info = env.step(actions_as_single)
        observations = batch_dict_of_items(observation)
        rewards = np.array([reward])
        dones = np.array([done])
        truncateds = np.array([truncated])
        infos = np.array([info])
    return observations, rewards, dones, truncateds, infos



def env_batch_reset(env: gym.Env) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reset the environment in batch mode.

    :param env: The environment.
    :type env: gym.Env

    :return: The batch of observations.
    :rtype: np.ndarray
    """
    if isinstance(env, gym.vector.VectorEnv):
        observations, infos = env.reset()
    else:
        observation, info = env.reset()
        observations = batch_dict_of_items(observation)
        infos = np.array([info])
    return observations, infos


def env_batch_render(env: gym.Env, **kwargs) -> List[Any]:
    """
    Render the environment in batch mode.

    :param env: The environment.
    :type env: gym.Env
    """
    if isinstance(env, gym.vector.VectorEnv):
        rendering = env.render()
        if rendering is None:
            rendering = [None for _ in range(env.num_envs)]
    else:
        rendering = env.render()
        rendering = [rendering]
    return rendering


def get_single_observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Return the observation space of a single environment.

    :param env: The environment.
    :type env: gym.Env

    :return: The observation space.
    :rtype: gym.spaces.Space
    """
    if isinstance(env, gym.vector.VectorEnv):
        return env.single_observation_space
    else:
        return env.observation_space


def get_single_action_space(env: gym.Env) -> gym.spaces.Space:
    """
    Return the action space of a single environment.

    :param env: The environment.
    :type env: gym.Env

    :return: The action space.
    :rtype: gym.spaces.Space
    """
    if isinstance(env, gym.vector.VectorEnv):
        return env.single_action_space
    else:
        return env.action_space


def sample_action_space(action_space: gym.spaces.Space, re_format: str = "raw"):
    """
    Sample an action from the action space.

    :param action_space: The action space.
    :type action_space: gym.spaces.Space
    :param re_format: The format to return the action in.
    :type re_format: str

    :return: The sampled action.
    :rtype: Any
    """
    if isinstance(action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete, gym.spaces.Box)):
        action = action_space.sample()
    else:
        raise NotImplementedError(f"Action space {action_space} is not implemented.")
    if re_format == "raw":
        return action
    elif re_format == "one_hot":
        if isinstance(action_space, gym.spaces.Discrete):
            return np.eye(action_space.n)[action]
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            return np.stack([np.eye(n)[a] for n, a in zip(action_space.nvec, action)])
        else:
            raise NotImplementedError(f"Action space {action_space} is not implemented.")
    else:
        raise NotImplementedError(f"Format {re_format} is not implemented.")


def discounted_cumulative_sums(x, discount, axis=-1, **kwargs):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    from scipy.signal import lfilter
    conv = lfilter([1], [1, float(-discount)], x[::-1], axis=axis)[::-1]
    return conv


def batch_numpy_actions(actions, env: Optional[gym.Env] = None):
    actions_as_numpy = to_numpy(actions)
    if isinstance(actions_as_numpy, np.ndarray):
        dim = len(actions_as_numpy.shape)
        if dim == 0:
            return np.expand_dims(actions_as_numpy, axis=0)
    elif isinstance(actions_as_numpy, dict):
        return {k: batch_numpy_actions(v) for k, v in actions_as_numpy.items()}
    else:
        raise NotImplementedError(f"Type {type(actions_as_numpy)} is not implemented.")
    if env is not None:
        actions_as_numpy = format_numpy_actions(actions_as_numpy, env)
    return actions_as_numpy


def format_numpy_actions(actions, env: gym.Env):
    actions_as_numpy = to_numpy(actions)
    entry_is_dict = isinstance(actions_as_numpy, dict)
    spec = space_to_spec(env.action_space)
    if not entry_is_dict:
        assert len(spec) == 1, f"Expected only one action space, but got {len(spec)}."
        actions_as_numpy = {k: actions_as_numpy for k in spec}
    for k, space in spec.items():
        if isinstance(space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
            actions_as_numpy[k] = actions_as_numpy[k].astype(np.int64)
        elif isinstance(space, gym.spaces.Box):
            actions_as_numpy[k] = actions_as_numpy[k].astype(np.float32)
        else:
            pass
    if not entry_is_dict:
        actions_as_numpy = actions_as_numpy[list(actions_as_numpy.keys())[0]]
    return actions_as_numpy


class TrajectoryRenderer:
    def __init__(
            self,
            trajectory: Trajectory,
            env: Optional[gym.Env] = None,
            **kwargs
    ):
        self.trajectory = trajectory
        self.env = env
        if self.check_simulate_is_needed():
            if self.env is None:
                raise ValueError(
                    "If the experiences in the trajectory do not have others['render'], an environment must be provided."
                )
            self.simulate()

    def check_simulate_is_needed(self):
        is_needed = not all("render" in x.others for x in self.trajectory)
        return is_needed

    def simulate(self):
        warnings.warn(
            "This method is deprecated. Its highly recommended to set the the attribute others['render'] "
            "when generating trajectories. If you are using RLAcademy.generate_trajectories, you can set "
            "the argument render=True to do so easily."
        )
        for x in self.trajectory:
            self.env.unwrapped.state = x.obs
            x.others["render"] = self.env.render()

    def render(self, **kwargs) -> Tuple[plt.Figure, plt.Axes, mpl_animation.FuncAnimation]:
        filename = kwargs.get("filename", None)
        file_extension = kwargs.get("file_extension", "gif")
        writer = kwargs.get("writer", "ffmpeg")
        fps = kwargs.get("fps", 30)
        time_interval = 1 / fps

        fig = kwargs.get("fig", None)
        ax = kwargs.get("ax", None)
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        env_name = self.env.unwrapped.spec.id
        title = f"Trajectory on {env_name}.\nCumulative reward: {self.trajectory.cumulative_reward:.2f}."
        title = kwargs.get("title", title)
        fig.suptitle(title, fontsize=kwargs.get("title_font_size", 16))
        ax.axis("off")
        im = ax.imshow(self.trajectory[0].others["render"])

        def _animation(i):
            im.set_array(self.trajectory[i].others["render"])
            return im,

        anim = mpl_animation.FuncAnimation(
            fig, _animation, frames=len(self.trajectory), interval=time_interval, blit=True
        )
        if filename is not None:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            if file_extension is None:
                if '.' in filename:
                    file_extension = filename.split('.')[-1]
                else:
                    file_extension = 'gif'
            assert file_extension in ["mp4", "gif"], "The extension of the file must be mp4 or gif."
            if filename.endswith(file_extension):
                filename = ''.join(filename.split('.')[:-1])
            anim.save(f"{filename}.{file_extension}", writer=writer, fps=fps)
        if kwargs.get("show", True):
            plt.show()
        return fig, ax, anim

    def to_file(self, file_path: str, fps: int = 30, **kwargs):
        import imageio

        if "." in file_path:
            ext = os.path.splitext(file_path)[-1]
        else:
            ext = kwargs.get("ext", ".mp4")
        if not file_path.endswith(ext):
            file_path += ext
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with imageio.get_writer(file_path, fps=fps) as writer:
            for x in self.trajectory:
                writer.append_data(x.others["render"])
        return file_path

    def to_mp4(self, file_path: str, fps: int = 30, **kwargs):
        return self.to_file(file_path, fps=fps, ext="mp4", **kwargs)

    def to_gif(self, file_path: str, fps: int = 30, **kwargs):
        return self.to_file(file_path, fps=fps, ext="gif", **kwargs)


def continuous_actions_distribution(
        actions: Union[Dict, torch.Tensor, np.ndarray],
        covariance: Optional[Union[Dict, torch.Tensor, np.ndarray]] = None
) -> Union[Dict, torch.distributions.Distribution]:
    """
    Creates a continuous action distribution from the actions and the covariance.

    :param actions: The actions.
    :type actions: Union[Dict, torch.Tensor, np.ndarray]
    :param covariance: The covariance of the actions. If None, a diagonal covariance is assumed using the variance of
        the given actions.
    :type covariance: Optional[Union[Dict, torch.Tensor, np.ndarray]]
    :return: The action distribution.
    :rtype: Union[Dict, torch.distributions.Distribution]
    """
    if isinstance(actions, dict):
        dist = {
            k: continuous_actions_distribution(actions[k], covariance[k] if covariance is not None else None)
            for k in actions
        }
    else:
        actions = to_tensor(actions)
        if covariance is None:
            std = torch.std(actions.view(-1, actions.shape[-1]), dim=0)
            covariance = torch.diag(std**2)
        else:
            covariance = to_tensor(covariance)
        dist = torch.distributions.MultivariateNormal(actions, covariance.to(actions.device))
    return dist


def batch_dict_of_items(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: batch_dict_of_items(v) for k, v in x.items()}
    else:
        return np.array([x])


def get_item_from_batch(x: Any, i: int) -> Any:
    if isinstance(x, dict):
        return {k: get_item_from_batch(v, i) for k, v in x.items()}
    else:
        return x[i]
