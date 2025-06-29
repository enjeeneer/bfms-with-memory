"""Module defining the replay buffer for the FB agent."""
from copy import deepcopy
from pathlib import Path
import torch
import numpy as np
from loguru import logger
from tqdm import tqdm

from rewards import RewardFunctionConstructor
from typing import Dict, Tuple, List, Union

from agents.base import OfflineReplayBuffer, Batch


class MemoryEfficientOfflineReplayBuffer:
    """
    Replay buffer. Performs frame stacking
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
