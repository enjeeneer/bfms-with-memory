# pylint: disable=[W0212, W0237, E1102, R1701, W0613]

"""Module for holding abstract base classes for all agents."""

import abc
import numpy as np
import torch
import wandb
from tqdm import tqdm
from loguru import logger
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

from agents.s4d import S4
from agents.transformer import Transformer, Cache, TransformerHiddenState
from agents.utils import TruncatedNormal, SquashedNormal
from rewards import RewardFunctionConstructor


class AbstractAgent(torch.nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for all agents."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        name: str,
    ):
        super().__init__()
        self._observation_dimension = observation_length
        self._action_dimension = action_length
        self._name = name

    @property
    def observation_length(self) -> int:
        """Length of observation space used as input to agent."""
        return self._observation_dimension

    @property
    def action_length(self) -> int:
        """Length of action space used as input to agent."""
        return self._action_dimension

    @property
    def name(self) -> str:
        """
        Agent name.
        """
        return self._name

    @abc.abstractmethod
    def act(self, *args, **kwargs) -> torch.Tensor:
        """
        Returns an action for a given input.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> Dict:
        """
        Updates parameters of model.
        """
        raise NotImplementedError

    def save(self, dir_path: Path) -> Path:
        """
        Saves a copy of the model in a format that can be loaded by load
        """
        dir_path.mkdir(exist_ok=True)
        save_path = dir_path / Path(str(self._name) + ".pickle")
        torch.save(self, save_path)

        return save_path

    @abc.abstractmethod
    def load(self, filepath: Path):
        pass


class AbstractMLP(torch.nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for all feedforward networks."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        hidden_dimension: int,
        hidden_layers: int,
        activation: str,
        device: torch.device,
        preprocessor: bool = False,
        layernorm: bool = False,
    ):
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension
        self._hidden_dimension = hidden_dimension
        self._hidden_layers = hidden_layers
        self._activation = activation
        self.device = device
        self._preprocessor = preprocessor
        self._layernorm = layernorm

        super().__init__()
        self.trunk = self._build()

    def _build(self) -> torch.nn.Sequential:
        """
        Creates MLP trunk.
        """
        if self.hidden_layers == 0:
            function = [torch.nn.Linear(self.input_dimension, self.output_dimension)]
        else:
            # first layer
            # ICLR paper uses layer norm and tanh for first layer of every network
            if self._layernorm:
                function = [
                    torch.nn.Linear(self.input_dimension, self.hidden_dimension),
                    torch.nn.LayerNorm(self.hidden_dimension),
                    torch.nn.Tanh(),
                ]
            else:
                function = [
                    torch.nn.Linear(self.input_dimension, self.hidden_dimension),
                    self.activation,
                ]

            # hidden layers
            for _ in range(self.hidden_layers - 1):
                function += [
                    torch.nn.Linear(self.hidden_dimension, self.hidden_dimension),
                    self.activation,
                ]

            # last layer
            function.append(
                torch.nn.Linear(self.hidden_dimension, self.output_dimension)
            )

        # add non-linearity to last layer for preprocessor
        if self.preprocessor:
            function.append(self.activation)

        trunk = torch.nn.Sequential(*function).to(self.device)

        return trunk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes input through network.
        Args:
            x: tensor of shape [batch_dim, input_dimension]
        Returns:
            x: tensor of shape [batch_dim, output_dimension]
        """
        return self.trunk.forward(x)

    @property
    def input_dimension(self) -> int:
        return self._input_dimension

    @property
    def output_dimension(self) -> int:
        return self._output_dimension

    @property
    def hidden_dimension(self) -> int:
        return self._hidden_dimension

    @property
    def hidden_layers(self) -> int:
        return self._hidden_layers

    @property
    def activation(self) -> torch.nn:
        if self._activation == "relu":
            return torch.nn.ReLU()
        else:
            raise NotImplementedError(f"{self._activation} not implemented.")

    @property
    def preprocessor(self) -> bool:
        return self._preprocessor


class AbstractPreprocessor(AbstractMLP, metaclass=abc.ABCMeta):
    """Preprocesses an observation concatenated with another variable
    into a feature space."""

    def __init__(
        self,
        observation_length: int,
        concatenated_variable_length: int,
        hidden_dimension: int,
        feature_space_dimension: int,
        hidden_layers: int,
        activation: str,
        device: torch.device,
        layernorm: bool = True,
    ):
        super().__init__(
            input_dimension=observation_length + concatenated_variable_length,
            output_dimension=feature_space_dimension,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
            preprocessor=True,
            layernorm=layernorm,
        )

    def forward(self, concatenation: torch.tensor) -> torch.tensor:
        """
        Passes concatenation through network to predict feature space
        Args:
            concatenation: tensor of shape
                        [batch_dim, observation_length + concatenated_variable_length]

        Returns:
            features: feature space tensor of shape [batch_dim, feature_space_dimension]
        """
        features = self.trunk(concatenation)  # pylint: disable=E1102

        return features


class AbstractCritic(AbstractMLP, metaclass=abc.ABCMeta):
    """Abstract critic class."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        hidden_dimension: int,
        hidden_layers: int,
        activation: str,
        device: torch.device,
        layernorm: bool = False,
        discrete: bool = False,
    ):
        self._observation_length = observation_length
        self._action_length = action_length
        self._hidden_dimension = hidden_dimension
        self._hidden_layers = hidden_layers

        if discrete:
            input_dimension = observation_length
            output_dimension = action_length
        else:
            input_dimension = observation_length + action_length
            output_dimension = int(1)

        super().__init__(
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
            preprocessor=False,
            layernorm=layernorm,
        )

    def forward(self, observation_action: torch.Tensor) -> torch.Tensor:
        """
        Passes observation_action pair through network to predict q value
        Args:
            observation_action: tensor of shape
                                        [batch_dim, observation_length + action_length]

        Returns:
            q: q value tensor of shape [batch_dim, 1]
        """
        q = self.trunk(observation_action)  # pylint: disable=E1102

        return q


class DoubleQCritic(torch.nn.Module):
    """Critic network employing double Q learning."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        hidden_dimension: int,
        hidden_layers: int,
        activation: str,
        device: torch.device,
        layernorm: bool = False,
        discrete: bool = False,
    ):
        super().__init__()

        self.Q1 = AbstractCritic(
            observation_length=observation_length,
            action_length=action_length,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
            layernorm=layernorm,
            discrete=discrete,
        )
        self.Q2 = AbstractCritic(
            observation_length=observation_length,
            action_length=action_length,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
            layernorm=layernorm,
            discrete=discrete,
        )
        self.outputs = {}
        self.discrete = discrete

    def forward(
        self, observation: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passes obs-action pair through q functions.
        Args:
            observation: tensor of shape [batch_dimension, observation_length]
            action: tensor of shape [batch_dimension, action_length]

        Returns:
            q1: q value from first q function
            q2: q value from second q function
        """

        if not self.discrete:
            assert observation.size(0) == action.size(0)
            x = torch.cat([observation, action], dim=-1)
        else:
            x = observation
        q1 = self.Q1.forward(x)
        q2 = self.Q2.forward(x)

        self.outputs["q1"] = q1
        self.outputs["q2"] = q2

        return q1, q2


class VCritic(torch.nn.Module):
    """
    State value function.
    """

    def __init__(
        self,
        observation_length: int,
        hidden_dimension: int,
        hidden_layers: int,
        activation: str,
        device: torch.device,
        layernorm: bool = False,
    ):
        super().__init__()

        self.V = AbstractCritic(
            observation_length=observation_length,
            action_length=0,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
            layernorm=layernorm,
        )
        self.outputs = {}

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passes obs-action pair through q functions.
        Args:
            observation: tensor of shape [batch_dimension, observation_length]
        Returns:
            v1: value from first v function
        """

        v = self.V.forward(observation)

        self.outputs["v"] = v

        return v


class DoubleVCritic(torch.nn.Module):
    """
    Double state value function.
    """

    def __init__(
        self,
        observation_length: int,
        hidden_dimension: int,
        hidden_layers: int,
        activation: str,
        device: torch.device,
        layernorm: bool = False,
    ):
        super().__init__()

        self.V1 = VCritic(
            observation_length=observation_length,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
            layernorm=layernorm,
        )
        self.V2 = VCritic(
            observation_length=observation_length,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
            layernorm=layernorm,
        )
        self.outputs = {}

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passes obs-action pair through q functions.
        Args:
            observation: tensor of shape [batch_dimension, observation_length]
        Returns:
            v1: value from first v function
        """

        v1 = self.V1.forward(observation)
        v2 = self.V2.forward(observation)

        self.outputs["v1"] = v1
        self.outputs["v2"] = v2

        return v1, v2


class AbstractActor(AbstractMLP, metaclass=abc.ABCMeta):
    """Abstract actor that selects action given input."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        hidden_dimension: int,
        hidden_layers: int,
        activation: str,
        device: torch.device,
        layernorms: bool,
    ):
        super().__init__(
            input_dimension=observation_length,
            output_dimension=action_length,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
            layernorm=layernorms,
        )

    def forward(
        self, observation: torch.Tensor, std: float
    ) -> torch.distributions.Distribution:
        """
        Passes input through network to predict action
        Args:
            observation: obs tensor of shape [batch_dim, input_length]
            std: standard deviation of action distribution
        Returns:
            action: action tensor of shape [batch_dim, action_length]
        """
        if observation.shape[-1] != self.input_dimension:
            raise ValueError(
                f"Input shape {observation.shape} does not "
                f"match input dimension {self.input_dimension}"
            )

        mu = self.trunk(observation)  # pylint: disable=E1102
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = TruncatedNormal(mu, std)

        return dist


class AbstractGaussianActor(AbstractMLP, metaclass=abc.ABCMeta):
    """Abstract gaussian actor that selects action given input."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        hidden_dimension: int,
        hidden_layers: int,
        activation: str,
        device: torch.device,
        log_std_bounds: Tuple[float] = (-5.0, 2.0),
        layernorm: bool = False,
    ):

        self.log_std_min = log_std_bounds[0]
        self.log_std_max = log_std_bounds[1]

        super().__init__(
            input_dimension=observation_length,
            output_dimension=action_length * 2,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
            layernorm=layernorm,
        )

    def forward(self, observation: torch.Tensor):
        """
        Takes observation and returns squashed normal distribution over action space.
        Args:
            observation: tensor of shape [batch_dim, observation_length]

        Returns:
            dist: SquashedNormal (multivariate Gaussian) dist over action space.

        """
        mu, log_std = self.trunk(observation).chunk(2, dim=-1)  # pylint: disable=E1102
        # output = self.trunk(observation)

        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )
        std = log_std.exp()
        dist = SquashedNormal(mu, std)

        return dist


class AbstractLogger(metaclass=abc.ABCMeta):
    """
    Abstract class for collecting metrics from training
    / eval runs.
    """

    def __init__(
        self, agent_config: Dict, use_wandb: bool = False, wandb_tags: List[str] = None
    ):
        self._agent_config = agent_config
        self.metrics = {}  # overwritten in concrete class

        if use_wandb:
            wandb.init(
                project="zero-shot",
                entity="zero-shot-rl",
                config=agent_config,
                tags=wandb_tags,
                reinit=True,
            )

    def log(self, metrics: Dict[str, float]):
        """Adds metrics to logger."""

        for key, value in metrics.items():
            try:
                self.metrics[key].append(value)
            except KeyError:
                raise KeyError(  # pylint: disable=W0707
                    f"Metric {key} not in metrics dictionary."
                )  # pylint: disable=W0707

        if wandb.run is not None:
            wandb.log(metrics)


@dataclasses.dataclass
class Batch:
    """
    Dataclass for batches of offline data.

    Args:
        observations: observations from current step in trajectory
        next_observations: observations from next step in trajectory
        other_observations: observations from anywhere in the dataset
        future_observations: observations from an arbitrary *future* step in trajectory
        discounts: future state discounts
        actions: actions from current step in trajectory
        rewards: rewards from transition
        observation_histories: history of observations including current timestep
        next_observation_histories: history of observations including next timestep
        positions: timestep index of first observation in history
        next_positions: timestep index of first observation in next history
        action_histories: history of actions including current timestep
        goal_histories: history of goals including current timestep for backward model
        next_goal_histories: history of goals including next timestep for backward model
        goal_action_histories: history of actions including current timestep
                                for backward model
        next_goal_action_histories: history of actions including next timestep
                                for backward model
        next_action_histories: history of next actions including next timestep
        not_dones: not done flags from transition
        physics: dm_control physics parameters
        goals: goal at current step in trajectory
        next_goals: goal at current step in trajectory
        future_goals: goals from an arbitrary *future* step in trajectory
        gciql_goals: goals for GC-IQL
    """

    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    discounts: Optional[torch.Tensor] = None
    not_dones: Optional[torch.Tensor] = None
    rewards: Optional[torch.Tensor] = None
    other_observations: Optional[torch.Tensor] = None
    future_observations: Optional[torch.Tensor] = None
    observation_histories: Optional[torch.Tensor] = None
    next_observation_histories: Optional[torch.Tensor] = None
    positions: Optional[torch.Tensor] = None
    next_positions: Optional[torch.Tensor] = None
    action_histories: Optional[torch.Tensor] = None
    next_action_histories: Optional[torch.Tensor] = None
    physics: Optional[torch.Tensor] = None
    goals: Optional[torch.Tensor] = None
    goal_histories: Optional[torch.Tensor] = None
    goal_action_histories: Optional[torch.Tensor] = None
    next_goal_action_histories: Optional[torch.Tensor] = None
    future_goal_action_histories: Optional[torch.Tensor] = None
    next_goal_histories: Optional[torch.Tensor] = None
    future_goal_histories: Optional[torch.Tensor] = None
    goal_positions: Optional[torch.Tensor] = None
    next_goal_positions: Optional[torch.Tensor] = None
    next_goals: Optional[torch.Tensor] = None
    future_goals: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    gciql_goals: Optional[torch.Tensor] = None

class AbstractReplayBuffer(metaclass=abc.ABCMeta):
    """
    Abstract replay buffer class for storing
    transitions from an environment.
    """

    def __init__(self, device: torch.device):
        self.device = device

    @abc.abstractmethod
    def add(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, batch_size: int) -> Dict:
        raise NotImplementedError


class AbstractOnlineReplayBuffer(AbstractReplayBuffer, metaclass=abc.ABCMeta):
    """Abstract buffer for online RL algorithms."""

    def __init__(
        self,
        capacity: int,
        observation_length: int,
        action_length: int,
        device: torch.device,
    ):
        super().__init__(device=device)
        self.observations = NotImplementedError("observations array not defined.")
        self.next_observations = NotImplementedError(
            "next_observations array not defined."
        )
        self.actions = NotImplementedError("actions array not defined.")
        self.rewards = NotImplementedError("rewards array not defined.")
        self.dones = NotImplementedError("dones array not defined.")
        self.current_memory_index = NotImplementedError(
            "current memory index not defined."
        )
        self.full_memory = NotImplementedError("full memory flag not implemented.")

        # properties
        self._capacity = capacity
        self._observation_length = observation_length
        self._action_length = action_length

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def observation_length(self) -> int:
        return self._observation_length

    @property
    def action_length(self) -> int:
        return self._action_length


class AbstractOfflineReplayBuffer(AbstractReplayBuffer, metaclass=abc.ABCMeta):
    """
    Abstract replay buffer class for storing
    transitions from an environment.
    """

    def __init__(self, device: torch.device):
        super().__init__(device)

        self.storage = NotImplementedError("Storage not implemented in base class.")

    @abc.abstractmethod
    def load_offline_dataset(
        self,
        *args,
        **kwargs,
    ) -> None:
        raise NotImplementedError


class MemoryEfficientOfflineReplayBuffer:
    """
    Replay buffer for pixel-based observations. Performs frame stacking
    and occlusion and sampling-time rather than loading-time to save
    memory.
    """

    def __init__(
        self,
        dataset_paths: List[Path],
        reward_constructors: Dict[Tuple[float, float], RewardFunctionConstructor],
        eval_multipliers: List[float],
        discount: float,
        device: torch.device,
        max_episodes: int,
        reward_occlusion: callable,
        dynamics_occlusion: callable,
        relabel: bool,
        frames: int,
        goal_frames: int,
        obs_type: str,
        pad_with_zeros: bool,
        task: str = None,
        load_on_init: bool = True,
        history_length: int = 0,
        goal_history_length: int = 0,
        future: float = 0.99,
        p_random_goal: float = 0.3,
        p_traj_goal: float = 0.5,
        p_curr_goal: float = 0.2,
        amago: bool = False,
    ):

        self._storage = {}
        self._reward_constructors = reward_constructors
        self._eval_multipliers = eval_multipliers
        self._max_episodes = max_episodes
        self._frames = frames
        self._goal_frames = goal_frames
        assert 0 <= future <= 1
        assert frames >= 1, "Frames must be >= 1"
        self._future = future
        self._episode_lengths = np.zeros(max_episodes, dtype=np.int32)
        self._idx = 0
        self._full = False
        self._discount = discount
        self._reward_occlusion = reward_occlusion
        self._dynamics_occlusion = dynamics_occlusion
        self._history_length = history_length
        self._goal_history_length = goal_history_length
        self._future = future
        self._obs_type = obs_type
        self._device = device
        self._is_fixed_episode_length = True
        self._episodes_selection_probability = None
        self._collected_episodes = 0
        self._num_transitions = 0
        self._p_random_goal = p_random_goal
        self._p_traj_goal = p_traj_goal
        self._relabel = relabel
        self._p_curr_goal = p_curr_goal
        self._pad_with_zeros = pad_with_zeros
        self._amago = amago
        if max_episodes is not None:
            self._episodes_per_dataset = int(max_episodes / len(dataset_paths))
        if reward_constructors is not None:
            env_keys = list(reward_constructors.keys())

        if load_on_init:
            datasets = []
            loaded_episodes = 0
            for path in dataset_paths:
                dataset, no_episodes = self.load_offline_dataset(
                    dataset_path=path,
                    relabel=relabel,
                    task=task,
                    reward_constructor=reward_constructors[env_keys[0]]
                    if reward_constructors
                    else None,
                )
                datasets.append(dataset)
                loaded_episodes += no_episodes

            # concatenate datasets
            self._storage = {}
            for key in datasets[0].keys():
                self._storage[key] = np.concatenate(
                    [dataset[key] for dataset in datasets]
                )
                print(f"{key} shape: {self._storage[key].shape}")
            self._full = True
            self._max_episodes = loaded_episodes

    def __len__(self) -> int:
        return self._max_episodes if self._full else self._idx

    def load_offline_dataset(
        self,
        dataset_path: Path,
        reward_constructor: RewardFunctionConstructor,
        relabel: bool,
        task: str,
    ):

        logger.info(f"Loading replay buffer from {dataset_path}")
        # load offline dataset in the form of episode paths
        episodes = np.load(dataset_path, allow_pickle=True)
        episodes = dict(episodes)

        storage = {}

        # load episodes
        for i, (_, episode) in enumerate(tqdm(episodes.items())):
            episode = episode.item()

            # relabel the episode
            if relabel:
                episode = self._relabel_episode(
                    episode=episode, task=task, reward_constructor=reward_constructor
                )

                # drop idx if nan exists in reward
                nan_idxs = np.where(np.isnan(episode["reward"]))[0]
                for key in episode.keys():
                    episode[key] = np.delete(episode[key], nan_idxs, axis=0)

            if i == self._episodes_per_dataset:
                break

            for name, values in episode.items():
                if name not in storage:
                    if name in ["observation"]:
                        storage[name] = np.empty(
                            (self._episodes_per_dataset,)
                            + (
                                values.shape[0],
                                self._dynamics_occlusion.observation_length,
                            ),
                            dtype=np.float32,
                        )
                    else:
                        storage[name] = np.empty(
                            (self._episodes_per_dataset,) + values.shape,
                            dtype=np.float32,
                        )
                    if "goal" not in storage:
                        storage["goal"] = np.empty(
                            (self._episodes_per_dataset,)
                            + (
                                values.shape[0],
                                self._reward_occlusion.observation_length,
                            ),
                            dtype=np.float32,
                        )
                # add dynamics and reward occlusion to observations
                if name == "observation":
                    storage["observation"][i] = self._dynamics_occlusion(
                        np.array(values, dtype=np.float32)
                    )
                    storage["goal"][i] = self._reward_occlusion(
                        np.array(values, dtype=np.float32)
                    )
                else:
                    storage[name][i] = np.array(values, dtype=np.float32)

                if name == "discount":
                    self._episode_lengths[self._idx] = (
                        len(values) - 1
                    )  # compensate for the dummy transition at the beginning

            # hack the dones (we know last transition is terminal)
            if "not_done" not in storage:
                storage["not_done"] = np.empty(
                    (self._episodes_per_dataset,)
                    + (self._episode_lengths[self._idx] + 1, 1),
                    dtype=float,
                )
            not_done = np.ones_like(storage["discount"][i], dtype=float)
            not_done[self._episode_lengths[self._idx]] = 0.0
            storage["not_done"][i] = not_done
            self._idx += 1

        no_episodes = storage["not_done"].shape[0]

        return storage, no_episodes

    @staticmethod
    def _relabel_episode(
        reward_constructor: RewardFunctionConstructor,
        episode: Dict[str, np.ndarray],
        task: str,
    ) -> np.array:
        """
        Takes episode data and relabels rewards w.r.t. the task.
        Args:
            reward_constructor: DMC environments (used for relabeling)
            episode: episode data
            task: task for reward relabeling
        Returns
            episode: the relabeled episode
        """

        env = reward_constructor.env
        task_idx = reward_constructor.task_names.index(task)
        episode = deepcopy(episode)

        rewards = []
        states = episode["physics"]

        # cycle through the states and relabel
        for i in range(states.shape[0]):
            try:
                with env.physics.reset_context():
                    env.physics.set_state(states[i])
                task_rewards = reward_constructor(env.physics)
                reward = np.full((1,), task_rewards[task_idx], dtype=np.float32)
                rewards.append(reward)
            except:  # pylint: disable=bare-except
                print(
                    "Error with physics during task inference dataset collection"
                    "Skipping timestep."
                )
                reward = np.full((1,), np.nan, dtype=np.float32)
                rewards.append(reward)

        episode["reward"] = np.array(rewards, dtype=np.float32)

        return episode

    def get_history(self, storage_key, ep_idx, step_idx, goal: bool = False):
        """
        Gets history of (stacked) observations.
        :param storage_key:
        :param ep_idx:
        :param step_idx:
        :param goal:
        :return:
        """
        if goal:
            history_length = self._goal_history_length
        else:
            history_length = self._history_length

        if history_length == 0:
            return None

        # Create a range of shifts for the history length
        history_offsets = np.arange(history_length) - (history_length - 1)

        rets = []
        for i in range(history_length):

            obs = self.get_obs(
                storage_key,
                ep_idx,
                np.maximum(0, step_idx - (history_length - i - 1)),
                goal=goal,
                frames=1,
            )
            rets.append(np.expand_dims(obs, 1))  # add history dim

        histories = np.concatenate(rets, axis=1)

        if self._pad_with_zeros:
            # Broadcast step_idx across the history length
            step_indices = step_idx[:, None] + history_offsets

            # get obs/action/pixel dimension
            dims = histories[0, 0, ...].shape[0]

            # Mask to find indices where step_idx < 0
            batch_mask = np.expand_dims(step_indices < 0, -1)
            mask = np.tile(batch_mask, (1, 1, dims))
            histories[mask] = 0

        return histories

    def get_obs(self, storage_key, ep_idx, step_idx, goal: bool = False, frames=None):
        """
        Creates stacked observations.
        :param storage_key:
        :param ep_idx:
        :param step_idx:
        :return:
        """
        if frames is None:
            if goal:
                frames = self._goal_frames
            else:
                frames = self._frames

        if frames == 1:
            if storage_key == "action":
                return self._storage[storage_key][ep_idx, step_idx]
            else:
                if goal:
                    return self._storage[storage_key][ep_idx, step_idx]
                else:
                    return self._storage[storage_key][ep_idx, step_idx]
        else:
            rets = []
            for i in range(frames):

                if storage_key == "action":
                    rets.append(
                        self._storage[storage_key][ep_idx, np.maximum(0, step_idx - i)]
                    )
                else:
                    if goal:
                        rets.append(
                            self._storage[storage_key][
                                ep_idx, np.maximum(0, step_idx - (frames - i - 1))
                            ]
                        )
                    else:
                        rets.append(
                            self._storage[storage_key][
                                ep_idx, np.maximum(0, step_idx - (frames - i - 1))
                            ]
                        )
            return np.stack(rets, axis=1)

    def sample(self, batch_size: int, return_rewards: bool = False) -> Union[Batch]:
        """
        Sample a batch of transitions from the replay buffer.
        """

        if self._is_fixed_episode_length:
            ep_idx = np.random.randint(0, len(self), size=batch_size)
            random_ep_idx = np.random.randint(0, len(self), size=batch_size)
        else:
            if self._episodes_selection_probability is None:
                self._episodes_selection_probability = (
                    self._episode_lengths / self._episode_lengths.sum()
                )
            ep_idx = np.random.choice(
                np.arange(len(self._episode_lengths)),
                size=batch_size,
                p=self._episodes_selection_probability,
            )
            random_ep_idx = np.random.choice(
                np.arange(len(self._episode_lengths)),
                size=batch_size,
                p=self._episodes_selection_probability,
            )

        eps_lengths = self._episode_lengths[ep_idx]
        random_eps_lengths = self._episode_lengths[random_ep_idx]
        # add +1 for the first dummy transition
        step_idx = (
            np.random.randint(0, eps_lengths - 1) + 1
        )  # -1 because of next action histories
        random_step_idx = np.random.randint(0, random_eps_lengths - 1) + 1
        assert (step_idx <= eps_lengths).all() and (
            random_step_idx <= random_eps_lengths
        ).all()
        if self._future < 1:
            future_idx = step_idx + np.random.geometric(
                p=(1 - self._future), size=batch_size
            )
            future_idx = np.clip(future_idx, 0, eps_lengths)
            assert (future_idx <= eps_lengths).all()

        if self._obs_type == "pixels":
            stacked_obs = self.get_obs("pixel", ep_idx, step_idx - 1)
            stacked_next_obs = self.get_obs("pixel", ep_idx, step_idx)
            stacked_goal = self.get_obs("pixel", ep_idx, step_idx - 1, goal=True)
            stacked_next_goal = self.get_obs("pixel", ep_idx, step_idx, goal=True)
            stacked_goal = stacked_goal.reshape(stacked_goal.shape[0], -1)
            stacked_next_goal = stacked_next_goal.reshape(
                stacked_next_goal.shape[0], -1
            )
        else:
            stacked_obs = self.get_obs("observation", ep_idx, step_idx - 1)
            stacked_next_obs = self.get_obs("observation", ep_idx, step_idx)
            stacked_goal = self.get_obs("goal", ep_idx, step_idx - 1, goal=True)
            stacked_next_goal = self.get_obs("goal", ep_idx, step_idx, goal=True)
            stacked_goal = stacked_goal.reshape(stacked_goal.shape[0], -1)
            stacked_next_goal = stacked_next_goal.reshape(
                stacked_next_goal.shape[0], -1
            )

        action = self._storage["action"][ep_idx, step_idx]

        # if we relabelled that means we want to sample rewards
        if return_rewards:
            reward = self._storage["reward"][ep_idx, step_idx]
        else:
            reward = None
        not_done = self._storage["not_done"][ep_idx, step_idx]
        discount = self._discount * self._storage["discount"][ep_idx, step_idx]
        stacked_future_obs = None
        if self._future < 1:
            if self._obs_type == "pixels":
                stacked_future_obs = self.get_obs("pixel", ep_idx, future_idx - 1)
                stacked_curr_obs = self.get_obs("pixel", ep_idx, step_idx - 1)
                stacked_future_goal = self.get_obs(
                    "pixel", ep_idx, future_idx - 1, goal=True
                )
                stacked_curr_goal = self.get_obs(
                    "pixel", ep_idx, step_idx - 1, goal=True
                )
            else:
                stacked_future_obs = self.get_obs("observation", ep_idx, future_idx - 1)
                stacked_curr_obs = self.get_obs("observation", ep_idx, step_idx - 1)
                stacked_future_goal = self.get_obs(
                    "goal", ep_idx, future_idx - 1, goal=True
                )
                stacked_curr_goal = self.get_obs(
                    "goal", ep_idx, step_idx - 1, goal=True
                )

            stacked_future_obs = np.where(
                (
                    np.random.rand(batch_size)
                    < self._p_curr_goal / (1.0 - self._p_random_goal)
                ).reshape(
                    stacked_future_obs.shape[0],
                    *[1] * (len(stacked_future_obs.shape) - 1),
                ),
                stacked_curr_obs,
                stacked_future_obs,
            )
            stacked_future_goal = np.where(
                (
                    np.random.rand(batch_size)
                    < self._p_curr_goal / (1.0 - self._p_random_goal)
                ).reshape(
                    stacked_future_goal.shape[0],
                    *[1] * (len(stacked_future_goal.shape) - 1),
                ),
                stacked_curr_goal,
                stacked_future_goal,
            )

            if self._obs_type == "pixels":
                stacked_random_obs = self.get_obs(
                    "pixel", random_ep_idx, random_step_idx - 1
                )
                stacked_random_goal = self.get_obs(
                    "pixel", random_ep_idx, random_step_idx - 1, goal=True
                )
            else:
                stacked_random_obs = self.get_obs(
                    "observation", random_ep_idx, random_step_idx - 1
                )
                stacked_random_goal = self.get_obs(
                    "goal", random_ep_idx, random_step_idx - 1, goal=True
                )
            stacked_future_obs = np.where(
                (np.random.rand(batch_size) < self._p_random_goal).reshape(
                    stacked_future_obs.shape[0],
                    *[1] * (len(stacked_future_obs.shape) - 1),
                ),
                stacked_random_obs,
                stacked_future_obs,
            )
            stacked_future_goal = np.where(
                (np.random.rand(batch_size) < self._p_random_goal).reshape(
                    stacked_future_goal.shape[0],
                    *[1] * (len(stacked_future_goal.shape) - 1),
                ),
                stacked_random_goal,
                stacked_future_goal,
            )

        # histories
        if self._history_length > 0:
            if self._obs_type == "pixels":
                stacked_obs_history = self.get_history("pixel", ep_idx, step_idx - 1)
                stacked_next_obs_history = self.get_history("pixel", ep_idx, step_idx)
            else:
                stacked_obs_history = self.get_history(
                    "observation", ep_idx, step_idx - 1
                )
                stacked_next_obs_history = self.get_history(
                    "observation", ep_idx, step_idx
                )

            action_history = self.get_history("action", ep_idx, step_idx)
            next_action_history = self.get_history("action", ep_idx, step_idx + 1)
            positions = np.expand_dims(step_idx, -1)
            next_positions = positions + 1
        else:
            obs_history = None
            next_obs_history = None
            action_history = None
            next_action_history = None
            positions = None
            next_positions = None

        if self._goal_history_length > 0:
            if self._obs_type == "pixels":
                goal_history = self.get_history(
                    "pixel", ep_idx, step_idx - 1, goal=True
                )
                next_goal_history = self.get_history(
                    "pixel", ep_idx, step_idx, goal=True
                )
                goal_history = torch.as_tensor(
                    goal_history,
                    device=self._device,
                )
                next_goal_history = torch.as_tensor(
                    next_goal_history,
                    device=self._device,
                )
                random_goal_history = self.get_history(
                    "pixel", random_ep_idx, random_step_idx - 1, goal=True
                )
                future_goal_history = self.get_history(
                    "pixel", ep_idx, future_idx - 1, goal=True
                )
                curr_goal_history = self.get_history(
                    "pixel", ep_idx, step_idx - 1, goal=True
                )
            else:
                goal_history = self.get_history("goal", ep_idx, step_idx - 1, goal=True)
                next_goal_history = self.get_history(
                    "goal", ep_idx, step_idx, goal=True
                )
                future_goal_history = self.get_history(
                    "goal", ep_idx, future_idx - 1, goal=True
                )
                curr_goal_history = self.get_history(
                    "goal", ep_idx, step_idx - 1, goal=True
                )
                goal_history = torch.as_tensor(
                    goal_history.reshape(  # pylint: disable=E1121
                        goal_history.shape[0], goal_history.shape[1], -1
                    ),
                    device=self._device,
                )
                next_goal_history = torch.as_tensor(
                    next_goal_history.reshape(  # pylint: disable=E1121
                        next_goal_history.shape[0], next_goal_history.shape[1], -1
                    ),
                    device=self._device,
                )
                random_goal_history = self.get_history(
                    "goal", random_ep_idx, random_step_idx - 1, goal=True
                )

            # actions are independent of obs type
            goal_action_history = self.get_history(
                "action", ep_idx, step_idx, goal=True
            )
            next_goal_action_history = self.get_history(
                "action", ep_idx, step_idx + 1, goal=True
            )
            future_goal_action_history = self.get_history(
                "action", ep_idx, future_idx - 1, goal=True
            )
            curr_goal_action_history = self.get_history(
                "action", ep_idx, step_idx - 1, goal=True
            )
            random_goal_action_history = self.get_history(
                "action", ep_idx, random_step_idx - 1, goal=True
            )

            # HILP weighting
            goal_curr_indices = (
                np.random.rand(batch_size)
                < self._p_curr_goal / (1.0 - self._p_random_goal)
            ).reshape(
                future_goal_history.shape[0],
                *[1] * (len(future_goal_history.shape) - 1),
            )
            goal_action_curr_indices = (
                np.random.rand(batch_size)
                < self._p_curr_goal / (1.0 - self._p_random_goal)
            ).reshape(
                future_goal_action_history.shape[0],
                *[1] * (len(future_goal_action_history.shape) - 1),
            )

            future_goal_history = np.where(
                goal_curr_indices,
                curr_goal_history,
                future_goal_history,
            )
            future_goal_action_history = np.where(
                goal_action_curr_indices,
                curr_goal_action_history,
                future_goal_action_history,
            )
            goal_action_random_indices = (
                np.random.rand(batch_size) < self._p_random_goal
            ).reshape(
                future_goal_history.shape[0],
                *[1] * (len(future_goal_action_history.shape) - 1),
            )
            future_goal_history = np.where(
                goal_action_random_indices, random_goal_history, future_goal_history
            )
            future_goal_action_history = np.where(
                goal_action_random_indices,
                random_goal_action_history,
                future_goal_action_history,
            )
            goal_action_history = torch.as_tensor(
                goal_action_history, device=self._device
            )
            next_goal_action_history = torch.as_tensor(
                next_goal_action_history, device=self._device
            )
            future_goal_history = torch.as_tensor(
                future_goal_history,
                device=self._device,
            )
            future_goal_action_history = torch.as_tensor(
                future_goal_action_history,
                device=self._device,
            )
            goal_positions = torch.as_tensor(
                np.expand_dims(step_idx, -1), dtype=torch.int, device=self._device
            )
            next_goal_positions = goal_positions + 1
        else:
            goal_history = None
            next_goal_history = None
            goal_positions = None
            next_goal_positions = None
            goal_action_history = None
            next_goal_action_history = None
            future_goal_history = None
            future_goal_action_history = None

        # if we're dealing with pixels then we flatten
        # all dims other than the first to get shape
        # [batch_size, channels * frames, height, width]
        if self._obs_type == "pixels":
            obs = torch.as_tensor(
                stacked_obs.reshape(
                    stacked_obs.shape[0],
                    -1,
                    stacked_obs.shape[-2],
                    stacked_obs.shape[-1],
                ),
                device=self._device,
            )
            next_obs = torch.as_tensor(
                stacked_next_obs.reshape(
                    stacked_next_obs.shape[0],
                    -1,
                    stacked_next_obs.shape[-2],
                    stacked_next_obs.shape[-1],
                ),
                device=self._device,
            )
            goal = torch.as_tensor(
                stacked_goal.reshape(
                    stacked_goal.shape[0],
                    -1,
                    stacked_goal.shape[-2],
                    stacked_goal.shape[-1],
                ),
                device=self._device,
            )
            next_goal = torch.as_tensor(
                stacked_next_goal.reshape(
                    stacked_next_goal.shape[0],
                    -1,
                    stacked_next_goal.shape[-2],
                    stacked_next_goal.shape[-1],
                ),
                device=self._device,
            )

        # if we're dealing with states then we flatten
        # all dims other than the first to get shape
        # [batch_size, obs_dim * frames]
        else:
            obs = torch.as_tensor(
                stacked_obs.reshape(stacked_obs.shape[0], -1), device=self._device
            )
            next_obs = torch.as_tensor(
                stacked_next_obs.reshape(stacked_next_obs.shape[0], -1),
                device=self._device,
            )
            goal = torch.as_tensor(
                stacked_goal.reshape(stacked_goal.shape[0], -1), device=self._device
            )
            next_goal = torch.as_tensor(
                stacked_next_goal.reshape(stacked_next_goal.shape[0], -1),
                device=self._device,
            )
        action = torch.as_tensor(action, device=self._device)
        if return_rewards:
            reward = torch.as_tensor(reward, device=self._device)
        discount = torch.as_tensor(discount, device=self._device)
        not_done = torch.as_tensor(not_done, device=self._device, dtype=torch.float32)

        if stacked_future_obs is not None:
            future_obs = torch.as_tensor(
                stacked_future_obs.reshape(stacked_future_obs.shape[0], -1),
                device=self._device,
            )
            future_goal = torch.as_tensor(
                stacked_future_goal.reshape(stacked_future_goal.shape[0], -1),
                device=self._device,
            )

        if self._history_length > 0:
            if self._obs_type == "pixels":
                obs_history = torch.as_tensor(
                    stacked_obs_history,
                    device=self._device,
                )
                next_obs_history = torch.as_tensor(
                    stacked_next_obs_history, device=self._device
                )

            else:
                obs_history = torch.as_tensor(
                    stacked_obs_history.reshape(  # pylint: disable=E1121
                        stacked_obs_history.shape[0], stacked_obs_history.shape[1], -1
                    ),
                    device=self._device,
                )
                next_obs_history = torch.as_tensor(
                    stacked_next_obs_history.reshape(  # pylint: disable=E1121
                        stacked_next_obs_history.shape[0],
                        stacked_next_obs_history.shape[1],
                        -1,
                    ),
                    device=self._device,
                )
            action_history = torch.as_tensor(action_history, device=self._device)
            next_action_history = torch.as_tensor(
                next_action_history, device=self._device
            )
            positions = torch.as_tensor(positions, device=self._device, dtype=torch.int)
            next_positions = torch.as_tensor(
                next_positions, device=self._device, dtype=torch.int
            )

        return Batch(
            observations=obs,
            actions=action,
            next_observations=next_obs,
            goals=goal,
            next_goals=next_goal,
            rewards=reward,
            discounts=discount,
            not_dones=not_done,
            future_observations=future_obs,
            future_goals=future_goal,
            future_goal_histories=future_goal_history,
            future_goal_action_histories=future_goal_action_history,
            observation_histories=obs_history,
            next_observation_histories=next_obs_history,
            positions=positions,
            next_positions=next_positions,
            action_histories=action_history,
            next_action_histories=next_action_history,
            goal_histories=goal_history,
            next_goal_histories=next_goal_history,
            goal_positions=goal_positions,
            next_goal_positions=next_goal_positions,
            goal_action_histories=goal_action_history,
            next_goal_action_histories=next_goal_action_history,
        )

    def sample_task_inference_transitions(
        self, inference_steps: int, popgym=False
    ) -> (
        Dict[Tuple[float, float], torch.Tensor],
        Dict[Tuple[float, float], Dict[str, torch.Tensor]],
        Dict[Tuple[float, float], torch.Tensor],
        Dict[Tuple[float, float], torch.Tensor],
    ):
        """
        Sample transitions from the replay buffer for FB task inference.
        Args:
            inference_steps: number of transitions to sample
        Returns:
            z_inf_observations: dictionary of task inference observations
                for each environment variant
            z_inf_goals: dictionary of reward-related states
                for each environment variant
            z_positions: dictionary of task inference rewards for each
                environment variant
            z_actions: dictionary of actions for each environment variant
        """

        if len(self._storage) == 0:
            raise RuntimeError(
                "The replay buffer is empty. Task inference sampling"
                "can only be performed after the replay buffer has been"
                "loaded."
            )

        # get indices
        ep_idx = np.random.randint(0, len(self), size=inference_steps)

        eps_lengths = self._episode_lengths[ep_idx]
        # add +1 for the first dummy transition
        step_idx = np.random.randint(0, eps_lengths) + 1

        # get observation and physics
        if self._obs_type == "pixels":
            batch_observations = self.get_obs("pixel", ep_idx, step_idx, goal=True)
        else:
            batch_observations = self.get_obs("goal", ep_idx, step_idx, goal=True)

        actions = self.get_obs("action", ep_idx, step_idx, goal=True)

        # if we're not using goal histories (i.e. not recurrent B) we flatten
        if self._goal_history_length > 0:
            goals = torch.as_tensor(batch_observations, device=self._device)
            actions = torch.as_tensor(actions, device=self._device)

        else:
            goals = torch.as_tensor(batch_observations, device=self._device).reshape(
                batch_observations.shape[0], -1
            )
            actions = torch.as_tensor(actions, device=self._device).reshape(
                actions.shape[0], -1
            )

        goal_positions = torch.as_tensor(
            np.expand_dims(step_idx, -1), dtype=torch.int, device=self._device
        )

        # process transitions for each environment variant
        z_inf_goals, z_inf_rewards, z_positions, z_actions = {}, {}, {}, {}

        for multiplier in self._eval_multipliers:
            # if we're not evaling on popgym then we
            # calculate rewards by resetting the env using the physics
            if not popgym:
                batch_physics = self._storage["physics"][ep_idx, step_idx]
                reward_constructor = self._reward_constructors[(multiplier, multiplier)]
                reward_dict = reward_constructor.process_timesteps(
                    batch_physics=batch_physics
                )
            else:
                rewards = torch.as_tensor(
                    self._storage["reward"][ep_idx, step_idx], device=self._device
                )
                reward_dict = {"default": rewards}

            z_inf_goals[(multiplier, multiplier)] = goals
            z_inf_rewards[(multiplier, multiplier)] = reward_dict
            z_positions[(multiplier, multiplier)] = goal_positions
            z_actions[(multiplier, multiplier)] = actions

        return z_inf_goals, z_inf_rewards, z_positions, z_actions

    def add(self, *args, **kwargs):
        pass


class OfflineReplayBuffer(AbstractOfflineReplayBuffer):
    """Offline replay buffer."""

    def __init__(
        self,
        reward_constructor: RewardFunctionConstructor,
        dataset_paths: List[Path],
        discount: float,
        device: torch.device,
        task: str,
        load_rewards: bool,
        dynamics_occlusion: callable,
        reward_occlusion: callable,
        frames: int,
        no_episodes: int = None,
        history_length: int = 0,
        relabel: bool = True,
        transitions: int = None,
        action_condition: dict = None,
        load_storage_on_init: bool = True,
        future: float = 0.99,
        p_random_goal: float = 0.3,
        p_traj_goal: float = 0.5,
        p_currgoal_goal: float = 0.2,
    ):
        super().__init__(device=device)

        self._discount = discount
        self._frames = frames
        self._future = future
        self._load_rewards = load_rewards
        self._reward_occlusion = reward_occlusion
        self._dynamics_occlusion = dynamics_occlusion
        self._number_of_datasets = len(dataset_paths)
        self._transitions_per_dataset = transitions // self._number_of_datasets
        self._history_length = history_length
        self._p_random_goal = p_random_goal
        self._p_traj_goal = p_traj_goal
        self._p_currgoal_goal = p_currgoal_goal

        assert transitions <= 1000 * no_episodes if no_episodes is not None else True

        # load dataset on init
        if load_storage_on_init:

            # load each dataset
            datasets = []
            for path in dataset_paths:
                dataset = self.load_offline_dataset(
                    reward_constructor=reward_constructor,
                    dataset_path=path,
                    relabel=relabel,
                    task=task,
                    action_condition=action_condition,
                    no_episodes=no_episodes,
                    transitions=self._transitions_per_dataset,
                )
                datasets.append(dataset)

            # concatenate datasets
            self.storage = {}
            for key in datasets[0].keys():
                if isinstance(datasets[0][key], torch.Tensor):
                    self.storage[key] = torch.cat(
                        [dataset[key] for dataset in datasets]
                    )
                elif isinstance(datasets[0][key], np.ndarray):
                    self.storage[key] = np.concatenate(
                        [dataset[key] for dataset in datasets], axis=0
                    )

    def load_offline_dataset(
        self,
        reward_constructor: RewardFunctionConstructor,
        dataset_path: Path,
        transitions: int,
        relabel: bool = True,
        task: str = None,
        action_condition: dict = None,
        rewards: bool = True,
        no_episodes: int = None,
    ) -> Dict:
        """
        Load the offline dataset into the replay buffer.
        Args:
            reward_constructor: DMC environments (used for relabeling)
            dataset_path: path to the dataset
            transitions: number of transitions to load from dataset
            relabel: whether to relabel the dataset
            task: task for reward relabeling
            action_condition: dict (action index: action value), we assume the
                            action index must always be higher than action value
            rewards: whether to load rewards
            no_episodes: number of episodes to load
        Returns:
            Dict: storage
        """

        storage = {}

        # load offline dataset in the form of episode paths
        episodes = np.load(dataset_path, allow_pickle=True)
        episodes = dict(episodes)

        observations = []
        goals = []
        actions = []
        rewards = []
        observation_histories = []
        positions = []
        next_observation_histories = []
        next_positions = []
        action_histories = []
        next_observations = []
        next_goals = []
        future_observations = []
        future_goals = []
        gciql_goals = []
        discounts = []
        not_dones = []
        physics = []

        # load the episodes
        for i, (_, episode) in enumerate(
            reversed(
                tqdm(episodes.items(), desc=f"Loading episodes from: {dataset_path}")
            )
        ):

            if no_episodes is not None and i >= no_episodes:
                break
            episode = episode.item()

            # relabel the episode
            if relabel:
                episode = self._relabel_episode(reward_constructor, episode, task)

            # store in lists
            observations.append(
                torch.as_tensor(
                    self._dynamics_occlusion(episode["observation"][:-1]),
                    device=self.device,
                )
            )
            goals.append(
                torch.as_tensor(
                    self._reward_occlusion(episode["observation"][:-1]),
                    device=self.device,
                )
            )
            actions.append(torch.as_tensor(episode["action"][1:], device=self.device))
            if self._load_rewards:
                rewards.append(
                    torch.as_tensor(episode["reward"][1:], device=self.device)
                )
            next_observations.append(
                torch.as_tensor(
                    self._dynamics_occlusion(episode["observation"][1:]),
                    device=self.device,
                )
            )
            next_goals.append(
                torch.as_tensor(
                    self._reward_occlusion(episode["observation"][1:]),
                    device=self.device,
                )
            )
            discounts.append(
                torch.as_tensor(
                    episode["discount"][1:] * self._discount, device=self.device
                )
            )
            physics.append(np.array(episode["physics"][:-1]))
            # hack the dones (we know last transition is terminal)
            not_done = torch.ones_like(
                torch.tensor(episode["discount"]), device=self.device
            )
            not_done[-1] = 0
            not_dones.append(not_done)

            # future observations
            # for each obs we randomly select an observation from the
            # future of the trajectory according to a geometric dist
            future_idxs = np.arange(len(episode["observation"]) - 1)
            future_idxs = future_idxs + np.random.geometric(
                p=(1 - self._future), size=len(episode["observation"]) - 1
            )
            future_idxs = np.clip(
                future_idxs, 0, len(episode["observation"]) - 2
            )  # 999
            future_observations.append(
                torch.as_tensor(
                    self._dynamics_occlusion(episode["observation"])[future_idxs],
                    device=self.device,
                )
            )
            future_goals.append(
                torch.as_tensor(
                    self._reward_occlusion(episode["observation"][future_idxs]),
                    device=self.device,
                )
            )

            # get gciql goals
            random_goal_idxs = np.random.randint(
                0, len(episode["observation"]) - 1, len(episode["observation"]) - 1
            )
            current_goal_idxs = np.arange(len(episode["observation"]) - 1)
            probs = np.random.random(len(episode["observation"]) - 1)
            gciql_goal_idxs = np.where(
                probs < self._p_traj_goal,
                future_idxs,
                np.where(
                    probs < (self._p_traj_goal + self._p_random_goal),
                    random_goal_idxs,
                    current_goal_idxs,
                ),
            )
            gciql_goals.append(
                torch.as_tensor(
                    episode["observation"][:-1][gciql_goal_idxs], device=self.device
                )
            )

            # create histories of observations and actions
            episode_observation_histories = []
            episode_action_histories = []
            episode_positions = []
            if self._history_length > 0:
                for i in range(len(episode["observation"])):

                    # if we are at the start of the episode and there is no full history
                    if i < (self._history_length - 1):
                        initial_observation_history = torch.as_tensor(
                            self._dynamics_occlusion(episode["observation"][: i + 1]),
                            device=self.device,
                        )
                        initial_action_history = torch.as_tensor(
                            episode["action"][: i + 1], device=self.device
                        )
                        episode_positions.append(
                            torch.as_tensor([i], device=self.device, dtype=torch.int)
                        )

                        # pad with zeros
                        episode_observation_histories.append(
                            torch.cat(
                                (
                                    torch.zeros(
                                        (
                                            self._history_length
                                            - initial_observation_history.shape[0],
                                            initial_observation_history.shape[-1],
                                        ),
                                        device=self.device,
                                    ),
                                    initial_observation_history,
                                ),
                                dim=0,
                            )
                        )
                        episode_action_histories.append(
                            torch.cat(
                                (
                                    torch.zeros(
                                        (
                                            self._history_length
                                            - initial_action_history.shape[0],
                                            initial_action_history.shape[-1],
                                        ),
                                        device=self.device,
                                    ),
                                    initial_action_history,
                                ),
                                dim=0,
                            )
                        )

                    else:
                        # add remaining observations and actions without padding
                        episode_observation_histories.append(
                            torch.as_tensor(
                                self._dynamics_occlusion(
                                    episode["observation"][
                                        i - (self._history_length - 1) : i + 1
                                    ]
                                ),
                                device=self.device,
                            )
                        )
                        episode_action_histories.append(
                            torch.as_tensor(
                                episode["action"][
                                    i - (self._history_length - 1) : i + 1
                                ],
                                device=self.device,
                            )
                        )
                        episode_positions.append(
                            torch.as_tensor([i], device=self.device, dtype=torch.int)
                        )

                # concatenate all histories in an episode
                observation_histories.append(
                    torch.stack(episode_observation_histories, dim=0)[:-1]
                )
                next_observation_histories.append(
                    torch.stack(episode_observation_histories, dim=0)[1:]
                )
                action_histories.append(
                    torch.stack(episode_action_histories, dim=0)[1:]
                )
                positions.append(torch.stack(episode_positions, dim=0)[:-1])
                next_positions.append(torch.stack(episode_positions, dim=0)[1:])

        # the below creates a "local" random number generator with fixed seed that
        # always subsamples the same transitions from the dataset, even if the
        # global seed is changed
        rng = np.random.default_rng(42)
        dataset_length = sum(len(obs) for obs in observations)

        logger.info(
            f"Sampling {transitions} transitions from"
            f" dataset of length {dataset_length}"
        )
        sample_indices = rng.choice(dataset_length, transitions, replace=False)

        # concatenate into storage
        storage["observations"] = torch.cat(observations)[sample_indices]
        storage["actions"] = torch.cat(actions)[sample_indices]
        if self._load_rewards:
            storage["rewards"] = torch.cat(rewards)[sample_indices]
        storage["next_observations"] = torch.cat(next_observations)[sample_indices]
        storage["goals"] = torch.cat(goals)[sample_indices]
        storage["next_goals"] = torch.cat(next_goals)[sample_indices]
        storage["future_goals"] = torch.cat(future_goals)[sample_indices]
        if self._history_length > 0:
            storage["observation_histories"] = torch.cat(observation_histories)[
                sample_indices
            ]
            storage["next_observation_histories"] = torch.cat(
                next_observation_histories
            )[sample_indices]
            storage["action_histories"] = torch.cat(action_histories)[sample_indices]
            storage["positions"] = torch.cat(positions)[sample_indices]
            storage["next_positions"] = torch.cat(next_positions)[sample_indices]

        # hilp future obs resampling
        # with probability self._random_goal we replace the future observation
        # with a random observation from the dataset
        future_observations = torch.cat(future_observations)
        if self._p_random_goal > 0:
            random_observations_idxs = torch.randperm(torch.cat(observations).shape[0])
            random_observations = torch.cat(observations)[random_observations_idxs]
            future_observations = torch.where(
                (
                    torch.rand(size=(future_observations.shape[0],), device=self.device)
                    < self._p_random_goal
                ).unsqueeze(-1),
                random_observations,
                future_observations,
            )

        storage["future_observations"] = future_observations[sample_indices]
        storage["discounts"] = torch.cat(discounts)[sample_indices]
        storage["physics"] = np.concatenate(physics)[sample_indices]
        storage["not_dones"] = torch.cat(not_dones)[sample_indices]
        storage["gciql_goals"] = torch.cat(gciql_goals)[sample_indices]

        # sub sample only the transitions that satisfy the action condition
        if action_condition is not None:
            for key, value in action_condition.items():
                action_condition_idxs = (
                    torch.where(storage["actions"][:, key] > value)[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                break

            storage["observations"] = storage["observations"][action_condition_idxs]
            storage["actions"] = storage["actions"][action_condition_idxs]
            if self._load_rewards:
                storage["rewards"] = storage["rewards"][action_condition_idxs]
            storage["next_observations"] = storage["next_observations"][
                action_condition_idxs
            ]
            storage["future_observations"] = storage["future_observations"][
                action_condition_idxs
            ]
            storage["discounts"] = storage["discounts"][action_condition_idxs]
            storage["physics"] = storage["physics"][action_condition_idxs]
            storage["not_dones"] = storage["not_dones"][action_condition_idxs]
            storage["goals"] = storage["goals"][action_condition_idxs]
            storage["next_goals"] = storage["next_goals"][action_condition_idxs]
            storage["future_goals"] = storage["future_goals"][action_condition_idxs]

        return storage

    @staticmethod
    def _relabel_episode(
        reward_constructor: RewardFunctionConstructor,
        episode: Dict[str, np.ndarray],
        task: str,
    ) -> np.array:
        """
        Takes episode data and relabels rewards w.r.t. the task.
        Args:
            reward_constructor: DMC environments (used for relabeling)
            episode: episode data
            task: task for reward relabeling
        Returns
            episode: the relabeled episode
        """

        env = reward_constructor.env
        task_idx = reward_constructor.task_names.index(task)
        episode = deepcopy(episode)

        rewards = []
        states = episode["physics"]

        # cycle through the states and relabel
        for i in range(states.shape[0]):
            with env.physics.reset_context():
                env.physics.set_state(states[i])
            task_rewards = reward_constructor(env.physics)
            reward = np.full((1,), task_rewards[task_idx], dtype=np.float32)
            rewards.append(reward)

        episode["reward"] = np.array(rewards, dtype=np.float32)

        return episode

    def sample(self, batch_size: int) -> Batch:
        """
        Samples OfflineBatch from the replay buffer.
        Args:
            batch_size: the batch size
        Returns:
            Batch: the batch of transitions
        """

        if len(self.storage) == 0:
            raise RuntimeError("The replay buffer is empty.")

        batch_indices = torch.randint(
            0, len(self.storage["observations"]), (batch_size,)
        )  # TODO: make attribute of replay buffer

        return Batch(
            observations=self.storage["observations"][batch_indices],
            actions=self.storage["actions"][batch_indices],
            rewards=self.storage["rewards"][batch_indices],
            observation_histories=self.storage["observation_histories"][batch_indices]
            if "observation_histories" in self.storage
            else None,
            next_observation_histories=self.storage["next_observation_histories"][
                batch_indices
            ]
            if "next_observation_histories" in self.storage
            else None,
            action_histories=self.storage["action_histories"][batch_indices]
            if "action_histories" in self.storage
            else None,
            positions=self.storage["positions"][batch_indices]
            if "positions" in self.storage
            else None,
            next_positions=self.storage["next_positions"][batch_indices]
            if "next_positions" in self.storage
            else None,
            next_observations=self.storage["next_observations"][batch_indices],
            future_observations=self.storage["future_observations"][batch_indices],
            discounts=self.storage["discounts"][batch_indices],
            not_dones=self.storage["not_dones"][batch_indices],
            physics=self.storage["physics"][batch_indices],
            goals=self.storage["goals"][batch_indices],
            next_goals=self.storage["next_goals"][batch_indices],
            future_goals=self.storage["future_goals"][batch_indices],
        )

    def add(self, *args, **kwargs):
        pass


class AbstractWorkspace(metaclass=abc.ABCMeta):
    """
    Abstract workspace for training and evaluating agents
    in an environment.
    """

    def __init__(self, reward_functions, env, wandb_logging: bool):
        self.reward_functions = reward_functions
        self.env = env
        self.wandb_logging = wandb_logging

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def eval(self, *args, **kwargs):
        raise NotImplementedError


class AbstractRecurrentEncoder(torch.nn.Module, metaclass=abc.ABCMeta):
    """Abstract class for recurrent networks that encode sequences."""

    def __init__(
        self,
        raw_input_dimension: int,  # dimension of raw sequence
        preprocessed_dimension: int,  # output dim of preprocessor
        postprocessed_dimension: int,  # output dim of postprocessor
        device: torch.device,
        transformer_dimension: Optional[
            int
        ] = None,  # output dim of transformer if used
        postprocessor=True,
    ):
        super().__init__()
        self._raw_input_dimension = raw_input_dimension
        self._preprocessed_dimension = preprocessed_dimension
        self._postprocessed_dimension = postprocessed_dimension
        self._transformer_dimension = transformer_dimension
        self._device = device
        self._postprocessor = postprocessor

        self.preprocessor = self._build_preprocessor()
        if postprocessor:
            self.postprocessor = self._build_postprocessor()

        self.encoder = NotImplementedError("Encoder not implemented in base class.")

    def _build_preprocessor(self) -> torch.nn.Sequential:
        """
        Builds the preprocessor. The preprocessor expands the
        input to a larger dimension and applies a layernorm
        and tanh activation for normalisation.
        """
        preprocessor_layers = []

        preprocessor_layers += [
            torch.nn.Linear(self._raw_input_dimension, self._preprocessed_dimension),
            torch.nn.LayerNorm(self._preprocessed_dimension),
            torch.nn.Tanh(),
            torch.nn.Linear(self._preprocessed_dimension, self._preprocessed_dimension),
            torch.nn.ReLU(),
        ]

        preprocessor = torch.nn.Sequential(*preprocessor_layers).to(self._device)

        return preprocessor

    def _build_postprocessor(self) -> torch.nn.Sequential:
        """
        Builds the postprocessor. The postprocessor reduces the dimension
        of the encoders output to the required output dimension and applies
        a ReLU activation.
        """

        if isinstance(self, (TransformerEncoder)):
            postprocessor_layer = [
                torch.nn.Linear(
                    self._transformer_dimension,
                    self._postprocessed_dimension,
                ),
                torch.nn.ReLU(),
            ]
        else:
            postprocessor_layer = [
                torch.nn.Linear(
                    self._preprocessed_dimension, self._postprocessed_dimension
                ),
                torch.nn.ReLU(),
            ]

        postprocessor = torch.nn.Sequential(*postprocessor_layer).to(self._device)

        return postprocessor

    @abc.abstractmethod
    def forward(self, **kwargs) -> torch.Tensor:
        """
        Encodes sequence by passing through the preprocessor,
        encoder, and postprocessor.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def init_internal_state(self) -> torch.Tensor:
        """
        Initializes the internal state of the recurrent encoder.
        """
        raise NotImplementedError


class MLPEncoder(AbstractRecurrentEncoder):
    """
    Encodes sequences using a straightforward MLP.
    """

    def __init__(
        self,
        raw_input_dimension: int,  # dimension of raw sequence
        preprocessed_dimension: int,  # output dim of preprocessor
        postprocessed_dimension: int,  # output dim of postprocessor
        layers: int,
        device: torch.device,
    ):
        super().__init__(
            raw_input_dimension=raw_input_dimension,
            preprocessed_dimension=preprocessed_dimension,
            postprocessed_dimension=postprocessed_dimension,
            device=device,
        )
        self.encoder = self._build_encoder(n_layers=layers)

    def _build_encoder(self, n_layers: int) -> torch.nn.Sequential:
        layers = []
        for _ in range(n_layers - 1):
            layers += [
                torch.nn.Linear(
                    self._preprocessed_dimension, self._preprocessed_dimension
                ),
                torch.nn.ReLU(),
            ]

        encoder = torch.nn.Sequential(*layers).to(self._device)

        return encoder

    def init_internal_state(self) -> None:
        """
        MLP encoder doesn't have internal state.
        """
        return None

    def forward(self, history: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, None]:
        """
        Passes history through the memory model.
        Args:
            history: context tensor of shape [batch_dim,
                    history_length, observation_length + action_length + z_dimension]
        Returns:
            output = torch.Tensor: memory embedding of shape [batch_dim,
                                                        feature_space_dimension]
        """

        # preprocessor
        history = self.preprocessor.forward(
            history.reshape(history.size(0), -1),
        )

        # encoder
        output = self.encoder.forward(history)

        # postprocess
        output = self.postprocessor.forward(
            output,
        )

        return output, None


class LSTMEncoder(AbstractRecurrentEncoder):
    """
    Encodes sequences using an LSTM.
    """

    def __init__(
        self,
        raw_input_dimension: int,  # dimension of raw sequence
        preprocessed_dimension: int,  # output dim of preprocessor
        postprocessed_dimension: int,  # output dim of postprocessor
        device: torch.device,
    ):
        super().__init__(
            raw_input_dimension=raw_input_dimension,
            preprocessed_dimension=preprocessed_dimension,
            postprocessed_dimension=postprocessed_dimension,
            device=device,
        )

        self.encoder = torch.nn.LSTM(
            input_size=self._preprocessed_dimension,
            hidden_size=self._preprocessed_dimension,
            num_layers=1,
            batch_first=True,
            device=self._device,
        )

    def forward(
        self,
        history: torch.Tensor,
        previous_hidden_state: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # preprocess sequence
        history = self.preprocessor.forward(
            history,
        )

        # encode sequence
        if previous_hidden_state is not None:
            output, (hidden, cell) = self.encoder.forward(
                input=history, hx=previous_hidden_state
            )
        else:
            output, (hidden, cell) = self.encoder.forward(input=history)

        # postprocess output
        output = self.postprocessor.forward(
            hidden[-1],
        )  # final hidden state
        internal_state = (hidden, cell)

        return output, internal_state

    @torch.no_grad()
    def init_internal_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initializes the internal state of the LSTM encoder.
        """

        hidden_state = torch.zeros(
            1,
            1,
            self._preprocessed_dimension,
            device=self._device,
        )
        cell_state = torch.zeros(
            1,
            1,
            self._preprocessed_dimension,
            device=self._device,
        )
        internal_state = (hidden_state, cell_state)

        return internal_state


class GRUEncoder(AbstractRecurrentEncoder):
    """
    Encodes sequences using a GRU.
    """

    def __init__(
        self,
        raw_input_dimension: int,  # raw sequence dims
        preprocessed_dimension: int,  # preprocessor embedding dim
        postprocessed_dimension: int,  # postprocessor embedding dim
        gru_dimension: int,
        device: torch.device,
        num_layers: int,
        postprocessor=True,
    ):
        super().__init__(
            raw_input_dimension=raw_input_dimension,
            preprocessed_dimension=preprocessed_dimension,
            postprocessed_dimension=postprocessed_dimension,
            device=device,
            postprocessor=postprocessor,
        )
        self.encoder = torch.nn.GRU(
            input_size=self._preprocessed_dimension,
            hidden_size=gru_dimension,
            num_layers=num_layers,
            batch_first=True,
            device=self._device,
        )
        self.num_layers = num_layers

    def forward(
        self,
        history: torch.Tensor,
        previous_hidden_state: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passes history through the memory model.
        Args:
            history: context tensor of shape [batch_dim,
                    history_length, observation_length + action_length + z_dimension]
            previous_hidden_state: hidden state from previous time step
        Returns:
            hidden_state = torch.Tensor: memory embedding of shape [batch_dim,
                                                        feature_space_dimension]
        """
        # preprocessor
        history = self.preprocessor.forward(history)

        # encoder
        if previous_hidden_state is not None:
            output, hidden_state = self.encoder.forward(
                input=history, hx=previous_hidden_state
            )
        else:
            output, hidden_state = self.encoder.forward(input=history)

        # postprocess
        if self._postprocessor:
            output = self.postprocessor.forward(
                hidden_state[-1],
            )  # final hidden state
        else:
            output = hidden_state[-1]

        return output, hidden_state

    @torch.no_grad()
    def init_internal_state(self) -> torch.Tensor:
        """
        Initializes the internal state of the GRU encoder.
        """

        internal_state = torch.zeros(
            self.num_layers,
            1,
            self._preprocessed_dimension,
            device=self._device,
        )

        return internal_state


class S4DEncoder(AbstractRecurrentEncoder):
    """
    Encodes sequences using an S4D.
    """

    def __init__(
        self,
        raw_input_dimension: int,
        preprocessed_dimension: int,
        postprocessed_dimension: int,
        s4_dimension: int,
        num_layers: int,
        device: torch.device,
    ):
        super().__init__(
            raw_input_dimension=raw_input_dimension,
            preprocessed_dimension=preprocessed_dimension,
            postprocessed_dimension=postprocessed_dimension,
            device=device,
        )
        self.s4_dim = s4_dimension
        self.num_layers = num_layers
        self.encoder = torch.nn.ModuleList(
            [
                S4(
                    d_model=preprocessed_dimension,
                    d_state=self.s4_dim,
                    mode="diag",
                    measure="diag-lin",
                    bidirectional=False,
                    disc="zoh",
                    real_type="exp",
                    transposed=False,
                ).to(self._device)
                for _ in range(num_layers)
            ]
        )
        for layer in self.encoder:
            layer.setup_step()  # Required to initialize some parts of S4

    def forward(
        self,
        history: torch.Tensor,
        previous_hidden_state: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Passes history through the memory model.
        Args:
            history: context tensor of shape [batch_dim,
                    history_length, observation_length + action_length + z_dimension]
            previous_hidden_state: hidden state from previous time step
        Returns:
            hidden_state = torch.Tensor: memory embedding of shape [batch_dim,
                                                        feature_space_dimension]
        """
        # preprocessor
        x = self.preprocessor.forward(history)

        # encoder
        if previous_hidden_state is not None:
            # S4d .step() requires history of shape [1, embedding_dim]
            # and state of shape [1, embedding_dim, s4d state dim]

            x = x.squeeze(0)
            assert x.shape == (1, self._preprocessed_dimension)
            assert previous_hidden_state[0].shape == (
                1,
                self._preprocessed_dimension,
                self.s4_dim / 2,  # divide by 2 as complex no.
            )

            hidden_states = previous_hidden_state

            for i, layer in enumerate(self.encoder):
                x, hidden_state = layer.step(u=x, state=hidden_states[i])
                hidden_states[i] = hidden_state
            # postprocess
            output = self.postprocessor.forward(
                x,
            )
        else:
            hidden_states = []
            for i, layer in enumerate(self.encoder):
                x, hidden_state = layer.forward(u=x)
                hidden_states.append(hidden_state)

            # postprocess
            output = self.postprocessor.forward(
                x[:, -1, :],
            )  # final state

        return output, hidden_states

    @torch.no_grad()
    def init_internal_state(self) -> List[torch.Tensor]:
        """
        Initializes the internal state of the S4D encoder.
        """
        states = []

        for layer in self.encoder:
            state = layer.default_state(1).squeeze(0)
            real, imag = state.real, state.imag
            internal_state = torch.complex(real, imag)
            states.append(internal_state.unsqueeze(0))

        #  shape [num_layers, s4_dim, s4_dim]
        return states


class TransformerEncoder(AbstractRecurrentEncoder):
    """
    Encodes sequences using a Transformer.
    """

    def __init__(
        self,
        raw_input_dimension: int,  # raw sequence dims
        preprocessed_dimension: int,  # preprocessor embedding dim
        postprocessed_dimension: int,  # postprocessor embedding dim
        transformer_dimension: int,  # k, v, q dim
        transformer_n_heads: int,
        num_layers: int,
        device: torch.device,
        pooling: str,
        transformer_attention: str,
        history_length: int,
    ):
        super().__init__(
            raw_input_dimension=raw_input_dimension,
            preprocessed_dimension=preprocessed_dimension,
            postprocessed_dimension=postprocessed_dimension,
            device=device,
            transformer_dimension=transformer_dimension,
        )

        self._history_length = history_length
        self._pooling = pooling
        self._device = device
        self._max_sequence_length = 1000 + history_length

        self.encoder = Transformer(
            inp_dim=preprocessed_dimension,
            max_pos_idx=self._max_sequence_length,
            d_model=transformer_dimension,
            n_heads=transformer_n_heads,
            layers=num_layers,
            attention=transformer_attention,
        ).to(self._device)

    def forward(
        self,
        history: torch.Tensor,
        first_time_idx: torch.Tensor,
        previous_hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # can be either history length (forward model) or history_length - 1 (actor)
        sequence_length = history.size(1)

        history = self.preprocessor.forward(history)
        time_idxs = first_time_idx + torch.arange(sequence_length).to(self._device)

        output, hidden_state = self.encoder.forward(
            seq=history,
            pos_idxs=time_idxs,
            hidden_state=previous_hidden_state,
        )

        output = self.postprocessor.forward(
            output,
        )

        # pool output
        if self._pooling == "last":
            output = output[:, -1, :]
        elif self._pooling == "mean":
            output = output.mean(dim=1)
        else:
            raise ValueError(f"Pooling method {self._pooling} not recognised.")

        return output, hidden_state

    def init_internal_state(self) -> TransformerHiddenState:
        """Initialises internal key, value, query states of transformer as Nones."""

        if self.encoder.attention == "vanilla":
            raise NotImplementedError(
                "Internal state initialisation not supported for vanilla attention."
            )

        assert (
            self.encoder.attention == "flash"
        ), "Internal state initialisation only supported for flash attention."

        def make_cache():
            return Cache(
                device=self._device,
                dtype=torch.bfloat16,
                batch_size=1,
                max_seq_len=self._max_sequence_length,
                n_heads=self.encoder.n_heads,
                head_dim=self.encoder.head_dim,
            )

        hidden_state = TransformerHiddenState(
            key_cache=[make_cache() for _ in range(self.encoder.n_layers)],
            val_cache=[make_cache() for _ in range(self.encoder.n_layers)],
            timesteps=torch.zeros((1,), dtype=torch.int32, device=self._device),
        )

        return hidden_state


class AbstractRecurrentActor(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    Defines an actor network that takes in a sequence of observations.
    """

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        postprocessed_dimension: int,
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        std_dev_clip: float,
        device: torch.device,
        obs_z_encoder: bool,
    ):
        super().__init__()

        self._observation_length = observation_length
        self._action_length = action_length
        self._std_dev_clip = std_dev_clip
        self._obs_z_encoder = obs_z_encoder

        self.obs_action_encoder = NotImplementedError(
            "Obs-action encoder not defined in base class."
        )
        self.obs_encoder = NotImplementedError(
            "Obs-z encoder not defined in base class."
        )
        self.positional_encoding = NotImplementedError(
            "Positional encoding not defined in base class."
        )

        self.output_layers = AbstractMLP(
            input_dimension=postprocessed_dimension * 2,
            output_dimension=action_length,
            hidden_dimension=actor_hidden_dimension,
            hidden_layers=actor_hidden_layers,
            activation="relu",
            device=device,
            layernorm=False,
        )

    def forward(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        std: float,
        sample: bool,
        z_history: Optional[torch.Tensor] = None,
        prev_hidden_obs_action_state: Optional[torch.Tensor] = None,
        prev_hidden_obs_z_state: Optional[torch.Tensor] = None,
        first_time_idx: Optional[torch.Tensor] = None,
        encoded_history: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the actor.
        Args:
            observation_history: tensor of shape
                                (batch_size, sequence_length, observation_length)
                                NB: the observations and actions are staggered
                                i.e [o_{t-L}, ... o_t] and [a_{t-L-1}, ... a_{t-1}]
            action_history: tensor of shape
                                (batch_size, sequence_length, action_length)
            z_history: tensor of shape
                                (batch_size, sequence_length, z_dimension)
                        NB: only passed if it is a zero-shot recurrent actor
            std: standard deviation to use for dist
            prev_hidden_obs_action_state (optional):
                        hidden state from previous forward pass
            prev_hidden_obs_z_state (optional): hidden state from previous forward pass
        Returns:
            dist: TruncatedNormal distribution over the action space
            hidden: hidden state of the GRU
        """
        # setup
        if z_history is not None:
            recurrent_input = torch.cat(
                [observation_history, action_history],
                dim=-1,
            )
            if self._obs_z_encoder:
                obs_encoder_input = torch.cat([observation_history, z_history], dim=-1)
            else:
                obs_encoder_input = torch.cat(
                    [observation_history[:, -1, :], z_history[:, -1, :]], dim=-1
                )
        else:
            recurrent_input = torch.cat([observation_history, action_history], dim=-1)
            if self._obs_z_encoder:
                obs_encoder_input = observation_history
            else:
                obs_encoder_input = observation_history[:, -1, :]

        if encoded_history is None:
            encoded_history, hidden_obs_action = self.obs_action_encoder(
                history=recurrent_input,
                first_time_idx=first_time_idx,
                previous_hidden_state=prev_hidden_obs_action_state,
            )
        else:
            hidden_obs_action = None

        # encode the single observation (with or without z)
        if self._obs_z_encoder:
            encoded_obs, hidden_obs_z = self.obs_encoder.forward(
                history=obs_encoder_input,
                first_time_idx=first_time_idx,
                previous_hidden_state=prev_hidden_obs_z_state,
            )
        else:
            encoded_obs = self.obs_encoder.forward(obs_encoder_input)
            hidden_obs_z = None

        # final layer converts the joint embedding to the mean of the distribution
        joint_encoding = torch.cat([encoded_history, encoded_obs], dim=-1)

        # pass through output layers to extract action
        mu = self.output_layers.forward(joint_encoding)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        action_dist = TruncatedNormal(mu, std)

        if sample:
            action = action_dist.sample(clip=self._std_dev_clip)
        else:
            action = action_dist.mean

        return action, hidden_obs_action, hidden_obs_z

    @abc.abstractmethod
    def init_internal_state(self):
        raise NotImplementedError


class GRUActor(AbstractRecurrentActor):
    """
    Actor that uses a GRU to encode the sequence of observations and actions.
    """

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        z_length: int,
        preprocessed_dimension: int,
        postprocessed_dimension: int,
        obs_encoder_hidden_dimension: int,
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        std_dev_clip: float,
        device: torch.device,
        obs_z_encoder: bool,
        num_layers: int,
        gru_dimension: int,
    ):
        super().__init__(
            observation_length=observation_length,
            action_length=action_length,
            postprocessed_dimension=postprocessed_dimension,
            actor_hidden_dimension=actor_hidden_dimension,
            actor_hidden_layers=actor_hidden_layers,
            std_dev_clip=std_dev_clip,
            device=device,
            obs_z_encoder=obs_z_encoder,
        )

        obs_action_input_dimension = observation_length + action_length

        self.obs_action_encoder = GRUEncoder(
            raw_input_dimension=obs_action_input_dimension,
            preprocessed_dimension=preprocessed_dimension,
            postprocessed_dimension=postprocessed_dimension,
            device=device,
            postprocessor=True,
            num_layers=num_layers,
            gru_dimension=gru_dimension,
        )
        # recurrent/non-recurrent encoder for obs-z
        if obs_z_encoder:
            self.obs_encoder = GRUEncoder(
                raw_input_dimension=observation_length + z_length,
                preprocessed_dimension=preprocessed_dimension,
                postprocessed_dimension=postprocessed_dimension,
                device=device,
                postprocessor=True,
                num_layers=num_layers,
                gru_dimension=gru_dimension,
            )
        else:
            self.obs_encoder = AbstractPreprocessor(
                observation_length=observation_length,
                concatenated_variable_length=z_length,
                feature_space_dimension=preprocessed_dimension,
                hidden_dimension=obs_encoder_hidden_dimension,
                hidden_layers=1,
                activation="relu",
                device=device,
                layernorm=True,
            )

        self._obs_z_encoder = obs_z_encoder

    def init_internal_state(
        self,
    ) -> Tuple[torch.Tensor, Optional[Union[torch.Tensor, None]]]:
        """
        Returns initial internal states of GRU encoder(s).
        """
        if self._obs_z_encoder:
            return (
                self.obs_action_encoder.init_internal_state(),
                self.obs_encoder.init_internal_state(),
            )
        else:
            return self.obs_action_encoder.init_internal_state(), None


class TransformerActor(AbstractRecurrentActor):
    """
    Actor that uses a Transformer to encode the sequence of observations and actions.
    """

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        z_length: int,
        preprocessed_dimension: int,
        postprocessed_dimension: int,
        transformer_dimension: int,
        transformer_n_heads: int,
        num_layers: int,
        pooling: str,
        history_length: int,
        obs_encoder_hidden_dimension: int,
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        transformer_attention: str,
        std_dev_clip: float,
        device: torch.device,
        obs_z_encoder: bool,
    ):
        super().__init__(
            observation_length=observation_length,
            action_length=action_length,
            postprocessed_dimension=postprocessed_dimension,
            actor_hidden_dimension=actor_hidden_dimension,
            actor_hidden_layers=actor_hidden_layers,
            std_dev_clip=std_dev_clip,
            device=device,
            obs_z_encoder=obs_z_encoder,
        )

        obs_action_input_dimension = observation_length + action_length

        self.obs_action_encoder = TransformerEncoder(
            raw_input_dimension=obs_action_input_dimension,
            preprocessed_dimension=preprocessed_dimension,
            postprocessed_dimension=postprocessed_dimension,
            transformer_dimension=transformer_dimension,
            transformer_n_heads=transformer_n_heads,
            num_layers=num_layers,
            device=device,
            pooling=pooling,
            history_length=history_length,
            transformer_attention=transformer_attention,
        )

        # recurrent/non-recurrent encoder for obs-z
        if obs_z_encoder:
            self.obs_encoder = TransformerEncoder(
                raw_input_dimension=observation_length + z_length,
                preprocessed_dimension=preprocessed_dimension,
                postprocessed_dimension=postprocessed_dimension,
                device=device,
                pooling=pooling,
                history_length=history_length,
                transformer_attention=transformer_attention,
                transformer_dimension=transformer_dimension,
                transformer_n_heads=transformer_n_heads,
                num_layers=num_layers,
            )
        else:
            self.obs_encoder = AbstractPreprocessor(
                observation_length=observation_length,
                concatenated_variable_length=z_length,
                feature_space_dimension=preprocessed_dimension,
                hidden_dimension=obs_encoder_hidden_dimension,
                hidden_layers=1,
                activation="relu",
                device=device,
                layernorm=True,
            )

        self._obs_z_encoder = obs_z_encoder

    def init_internal_state(
        self,
    ) -> Tuple[torch.Tensor, Optional[Union[torch.Tensor, None]]]:
        """
        Returns initial internal states of Transformer encoder(s).
        """
        if self._obs_z_encoder:
            return (
                self.obs_action_encoder.init_internal_state(),
                self.obs_encoder.init_internal_state(),
            )
        else:
            return self.obs_action_encoder.init_internal_state(), None


class S4DActor(AbstractRecurrentActor):
    """
    Actor that uses a S4D to encode the sequence of observations and actions.
    """

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        z_length: int,
        preprocessed_dimension: int,
        postprocessed_dimension: int,
        s4_dimension: int,
        obs_encoder_hidden_dimension: int,
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        std_dev_clip: float,
        device: torch.device,
        num_layers: int,
        obs_z_encoder: bool,
    ):
        super().__init__(
            observation_length=observation_length,
            action_length=action_length,
            postprocessed_dimension=postprocessed_dimension,
            actor_hidden_dimension=actor_hidden_dimension,
            actor_hidden_layers=actor_hidden_layers,
            std_dev_clip=std_dev_clip,
            device=device,
            obs_z_encoder=obs_z_encoder,
        )

        obs_action_input_dimension = observation_length + action_length

        self.obs_action_encoder = S4DEncoder(
            raw_input_dimension=obs_action_input_dimension,
            preprocessed_dimension=preprocessed_dimension,
            postprocessed_dimension=postprocessed_dimension,
            s4_dimension=s4_dimension,
            device=device,
            num_layers=num_layers,
        )

        # recurrent/non-recurrent encoder for obs-z
        if obs_z_encoder:
            self.obs_encoder = S4DEncoder(
                raw_input_dimension=observation_length + z_length,
                preprocessed_dimension=preprocessed_dimension,
                postprocessed_dimension=postprocessed_dimension,
                s4_dimension=s4_dimension,
                device=device,
                num_layers=num_layers,
            )
        else:
            self.obs_encoder = AbstractPreprocessor(
                observation_length=observation_length,
                concatenated_variable_length=z_length,
                feature_space_dimension=preprocessed_dimension,
                hidden_dimension=obs_encoder_hidden_dimension,
                hidden_layers=1,
                activation="relu",
                device=device,
                layernorm=True,
            )
        self._obs_z_encoder = obs_z_encoder

    def init_internal_state(
        self,
    ) -> Tuple[torch.Tensor, Optional[Union[torch.Tensor, None]]]:
        """
        Returns initial internal states of S4d encoder(s).
        """
        if self._obs_z_encoder:
            return (
                self.obs_action_encoder.init_internal_state(),
                self.obs_encoder.init_internal_state(),
            )
        else:
            return self.obs_action_encoder.init_internal_state(), None


class LSTMActor(AbstractRecurrentActor):
    """
    Actor that uses a LSTM to encode the sequence of observations and actions.
    """

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        z_length: int,
        preprocessed_dimension: int,  # preprocessor embedding dim
        postprocessed_dimension: int,  # postprocessor embedding dim
        obs_encoder_hidden_dimension: int,
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        std_dev_clip: float,
        device: torch.device,
        obs_z_encoder: bool,
    ):
        super().__init__(
            observation_length=observation_length,
            action_length=action_length,
            postprocessed_dimension=postprocessed_dimension,
            actor_hidden_dimension=actor_hidden_dimension,
            actor_hidden_layers=actor_hidden_layers,
            std_dev_clip=std_dev_clip,
            device=device,
            obs_z_encoder=obs_z_encoder,
        )

        obs_action_input_dimension = observation_length + action_length

        self.obs_action_encoder = LSTMEncoder(
            raw_input_dimension=obs_action_input_dimension,
            preprocessed_dimension=preprocessed_dimension,
            postprocessed_dimension=postprocessed_dimension,
            device=device,
        )

        # recurrent/non-recurrent encoder for obs-z
        if obs_z_encoder:
            self.obs_encoder = LSTMEncoder(
                raw_input_dimension=observation_length + z_length,
                preprocessed_dimension=preprocessed_dimension,
                postprocessed_dimension=postprocessed_dimension,
                device=device,
            )
        else:
            self.obs_encoder = AbstractPreprocessor(
                observation_length=observation_length,
                concatenated_variable_length=z_length,
                feature_space_dimension=preprocessed_dimension,
                hidden_dimension=obs_encoder_hidden_dimension,
                hidden_layers=1,
                activation="relu",
                device=device,
                layernorm=True,
            )
        self._obs_z_encoder = obs_z_encoder

    @torch.no_grad()
    def init_internal_state(
        self,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        Optional[Union[Tuple[torch.Tensor, torch.Tensor], None]],
    ]:
        """
        Initializes the internal state of the LSTM encoder.
        """
        if self._obs_z_encoder:
            return (
                self.obs_action_encoder.init_internal_state(),
                self.obs_encoder.init_internal_state(),
            )
        else:
            return self.obs_action_encoder.init_internal_state(), None


class MLPActor(AbstractRecurrentActor):
    """
    Actor that uses a MLP to encode the sequence of observations and actions.
    """

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        encoder_layers: int,
        preprocessed_dimension: int,  # preprocessor embedding dim
        postprocessed_dimension: int,  # postprocessor embedding dim
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        obs_encoder_hidden_dimension: int,
        history_length: int,
        std_dev_clip: float,
        device: torch.device,
        obs_z_encoder: bool,
        z_length: int,
    ):
        super().__init__(
            observation_length=observation_length,
            action_length=action_length,
            postprocessed_dimension=postprocessed_dimension,
            actor_hidden_dimension=actor_hidden_dimension,
            actor_hidden_layers=actor_hidden_layers,
            std_dev_clip=std_dev_clip,
            device=device,
            obs_z_encoder=obs_z_encoder,
        )

        obs_action_input_dimension = (observation_length + action_length) * (
            history_length
        )

        self.obs_action_encoder = MLPEncoder(
            raw_input_dimension=obs_action_input_dimension,
            preprocessed_dimension=preprocessed_dimension,
            postprocessed_dimension=postprocessed_dimension,
            layers=encoder_layers,
            device=device,
        )

        # recurrent/non-recurrent encoder for obs-z
        if obs_z_encoder:
            obs_z_input_dimension = (observation_length + z_length) * (history_length)
            self.obs_encoder = MLPEncoder(
                raw_input_dimension=obs_z_input_dimension,
                preprocessed_dimension=preprocessed_dimension,
                postprocessed_dimension=postprocessed_dimension,
                layers=encoder_layers,
                device=device,
            )
        else:
            self.obs_encoder = AbstractPreprocessor(
                observation_length=observation_length,
                concatenated_variable_length=z_length,
                feature_space_dimension=preprocessed_dimension,
                hidden_dimension=obs_encoder_hidden_dimension,
                hidden_layers=1,
                activation="relu",
                device=device,
                layernorm=True,
            )
        self._obs_z_encoder = obs_z_encoder

    @torch.no_grad()
    def init_internal_state(self) -> Tuple[None, None]:
        """
        MLP does not have internal state.
        """
        return None, None


class PopArtLayer(torch.nn.Module):
    """Popart"""

    def __init__(
        self,
        gammas: int,
        beta: float = 5e-4,
        init_nu: float = 100.0,
        enabled: bool = True,
    ):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.zeros(gammas, 1), requires_grad=False)
        self.nu = torch.nn.Parameter(
            torch.ones(gammas, 1) * init_nu, requires_grad=False
        )
        self.beta = beta
        self.w = torch.nn.Parameter(torch.ones((gammas, 1)), requires_grad=False)
        self.b = torch.nn.Parameter(torch.zeros((gammas, 1)), requires_grad=False)
        self._t = torch.nn.Parameter(torch.ones((gammas, 1)), requires_grad=False)
        self.enabled = enabled

    @property
    def sigma(self):
        return (torch.sqrt(self.nu - self.mu**2)).clamp(1e-4, 1e6)

    def normalize_values(self, val):
        if not self.enabled:
            return val
        return ((val - self.mu) / self.sigma).to(val.dtype)

    def to(self, device):
        self.w = self.w.to(device)
        self.b = self.b.to(device)
        self.mu = self.mu.to(device)
        self.nu = self.nu.to(device)
        return self

    def update_stats(self, val, mask):
        if not self.enabled:
            return
        assert val.shape == mask.shape
        self._t += 1
        old_sigma = self.sigma.data.clone()
        old_mu = self.mu.data.clone()
        # Use adaptive step size to reduce reliance on initialization (pg 13)
        beta_t = self.beta / (1.0 - (1.0 - self.beta) ** self._t)
        # dims are Batch, Length, 1, Gammas, 1
        total = mask.sum((0, 1, 2))
        mean = (val * mask).sum((0, 1, 2)) / total
        square_mean = ((val * mask) ** 2).sum((0, 1, 2)) / total
        self.mu.data = (1.0 - beta_t) * self.mu + beta_t * mean
        self.nu.data = (1.0 - beta_t) * self.nu + beta_t * square_mean
        self.w.data *= old_sigma / self.sigma
        self.b.data = ((old_sigma * self.b) + old_mu - self.mu) / (self.sigma)

    def forward(self, x, normalized=True):
        if not self.enabled:
            return x
        normalized_out = (self.w * x) + self.b
        if normalized:
            return normalized_out.to(x.dtype)
        else:
            return ((self.sigma * normalized_out) + self.mu).to(x.dtype)


def weight_init(m) -> None:
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            # if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            # if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class ImageEncoder(torch.nn.Module):
    """
    Simple CNN encoder for pixel observations.
    """

    def __init__(self, obs_shape) -> None:
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = None  # To be specified later

        if obs_shape[1] >= 64:
            self.convnet = torch.nn.Sequential(
                torch.nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 32, 3, stride=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 32, 3, stride=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 32, 3, stride=1),
                torch.nn.ReLU(),
            )
        elif obs_shape[1] >= 48:
            self.convnet = torch.nn.Sequential(
                torch.nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 32, 3, stride=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 32, 3, stride=1),
                torch.nn.ReLU(),
            )
        else:
            self.convnet = torch.nn.Sequential(
                torch.nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 32, 3, stride=1),
                torch.nn.ReLU(),
            )

        self.apply(weight_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class RandomShiftsAug(torch.nn.Module):
    """
    Shift augmentation for images.
    """

    def __init__(self, pad) -> None:
        super().__init__()
        self.pad = pad

    def forward(self, x) -> torch.Tensor:
        x = x.float()
        n, _, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = torch.torch.nn.functional.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return torch.torch.nn.functional.grid_sample(
            x, grid, padding_mode="zeros", align_corners=False
        )


def make_aug_encoder(image_wh, obs_shape, device):
    aug = RandomShiftsAug(pad=image_wh // 21)
    encoder = ImageEncoder(obs_shape).to(device)
    example_ob = torch.zeros(1, *obs_shape, device=device)
    module_obs_dim = encoder(example_ob).shape[-1]
    encoder.repr_dim = module_obs_dim
    return aug, encoder
