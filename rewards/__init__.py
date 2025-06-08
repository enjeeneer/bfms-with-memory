"""Module for constructing reward functions."""
import importlib
import torch
from typing import List, Dict, Tuple
import numpy as np
import custom_dmc_tasks as cdmc
import dmc
from utils import set_seed_everywhere
from loguru import logger
from tqdm import tqdm

from custom_dmc_tasks.walker import _DEFAULT_TIME_LIMIT as walker_time_limit
from custom_dmc_tasks.point_mass_maze import (
    _DEFAULT_TIME_LIMIT as point_mass_time_limit,
)
from custom_dmc_tasks.quadruped import _DEFAULT_TIME_LIMIT as quadruped_time_limit
from custom_dmc_tasks.cheetah import _DEFAULT_TIME_LIMIT as cheetah_time_limit


# NOTE: These default tasks aren't actually used;
# we just need to specify one to get the env to load
DEFAULT_TASKS = {
    "walker": "flip",
    "point_mass_maze": "reach_bottom_left_0",
    "cheetah": "run_backward",
    "quadruped": "stand",
    "jaco": "reach_bottom_left",
}

DEFAULT_TIME_LIMITS = {
    "walker": walker_time_limit,
    "point_mass_maze": point_mass_time_limit,
    "cheetah": cheetah_time_limit,
    "quadruped": quadruped_time_limit,
}


class RewardFunctionConstructor:
    """
    Given a domain and tasks, constructs a set of reward functions.
    """

    def __init__(
        self,
        domain_name: str,
        task_names: List[str],
        seed: int,
        obs_type: str,
        device: torch.device,
        frames: int,
        hardcode_seed: int = None,
        body_mass_multiplier: float = 1.0,
        damping_multiplier: float = 1.0,
        episode_length_multiplier: float = 1.0,
    ):
        if hardcode_seed is not None:
            set_seed_everywhere(hardcode_seed)

        self.domain_name = domain_name

        if "jaco" in domain_name:
            self.env = dmc.make(name="jaco_reach_bottom_left")
        else:
            self.env = cdmc.make(
                domain=domain_name,
                obs_type=obs_type,
                task=DEFAULT_TASKS[domain_name],
                environment_kwargs={"flat_observation": True},
                frames=frames,
                task_kwargs={
                    "random": seed,
                    "time_limit": int(
                        DEFAULT_TIME_LIMITS[domain_name] * episode_length_multiplier
                    ),
                },
            )

        # update physics with multipliers
        for i in range(self.env.physics.named.model.body_mass.shape[0]):
            self.env.physics.named.model.body_mass[i] = (
                self.env.physics.named.model.body_mass[i] * body_mass_multiplier
            )
        for j in range(self.env.physics.named.model.dof_damping.shape[0]):
            self.env.physics.named.model.dof_damping[j] = (
                self.env.physics.named.model.dof_damping[j] * damping_multiplier
            )

        self.task_names = task_names
        self.device = device
        self.reward_functions = {}
        if domain_name == "point_mass_maze":
            self.reward_functions = importlib.import_module(
                "rewards.point_mass_maze.multi_goal"
            ).reward_functions
            if "multi_goal" in task_names:
                self.task_names = list(self.reward_functions.keys())
            else:
                self.task_names = task_names

        else:
            for task in task_names:
                self.reward_functions[task] = importlib.import_module(
                    f"rewards.{domain_name}.{task}"
                ).reward_function

    def __call__(self, physics):
        return [self.reward_functions[task](physics) for task in self.task_names]

    def process_episode(
        self, episode: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict]:
        """Given an episode from an offline dataset, return observations and rewards
        for all tasks."""
        observations, rewards = [], []

        logger.info("Obtain reward-labelled states for task inference.")
        for physics, action in tqdm(
            zip(episode["physics"][:-1], episode["action"][1:])
        ):
            self.env.physics.set_state(physics)
            assert (physics == self.env.physics.get_state()).all()
            timestep = self.env.step(action)
            observations.append(timestep.observation["observations"])
            rewards.append(self(self.env.physics))

        rewards_dict = {}
        for i, task in enumerate(self.task_names):
            rewards_dict[task] = np.array(rewards)[:, i]

        return np.array(observations), rewards_dict

    def get_metadata(self):
        """
        Gets the environment metadata (e.g. observation length) for
        constructing agents.
        Args:
            None
        Returns:
            observation_spec: length of the observation space
            action_length: length of the action space
            action_range: range of the action space
        """
        if self.domain_name in ("jaco", "jaco_occluded"):
            observation_spec = self.env.observation_spec()
            action_length = self.env.action_spec().shape[
                0
            ]  # pylint: disable=protected-access
            action_range = [-1.0, 1.0]
        else:
            observation_spec = self.env.observation_spec()
            action_length = self.env.action_spec().shape[
                0
            ]  # pylint: disable=protected-access
            action_range = [
                self.env.action_spec().minimum[0],  # pylint: disable=protected-access
                self.env.action_spec().maximum[0],  # pylint: disable=protected-access
            ]

        return observation_spec, action_length, action_range

    def process_timesteps(self, batch_physics: np.ndarray) -> Dict[str, torch.tensor]:
        """
        Given a set of timesteps from an offline dataset, return observations and
        rewards for all tasks.
        """
        rewards = []

        # reset the env physics then pass through
        # reward function to get the reward
        for physics in batch_physics:
            with self.env.physics.reset_context():
                self.env.physics.set_state(physics)
            assert np.allclose(physics, self.env.physics.get_state(), atol=1e-5)
            rewards.append(self(self.env.physics))

        rewards = np.array(rewards)

        rewards_dict = {}
        for i, task in enumerate(self.task_names):
            rewards_dict[task] = torch.tensor(
                rewards, dtype=torch.float32, device=self.device
            )[:, i].unsqueeze(-1)

        return rewards_dict

    def rollout_agent(
        self, agent, zs: Dict, obs_flag: bool = False
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Given an agent and z vector, rollout an episode and return rewards and
        (potentially) observations.
        Args:
            agent: Agent for rollout
            zs: Dictionary mapping task names to z vectors
            obs_flag: Whether to return observations
        Returns:
            rewards: Dictionary mapping task names to rewards
            observations: Dictionary mapping task names to observations
        """
        rewards = {}
        observations = {}
        actions = {}
        # loop through zs specified for each task
        for task, z in zs.items():
            assert task in self.task_names, f"Env not built for task name: {task}"
            task_rewards = 0.0
            obs_list = []
            action_list = []
            reward_list = []

            timestep = self.env.reset()
            obs_list.append(timestep.observation["observations"])

            while not timestep.last():
                action, _ = agent.act(
                    timestep.observation["observations"],
                    task=z,
                    step=None,
                    sample=False,
                )
                timestep = self.env.step(action)
                task_rewards += self.reward_functions[task](self.env.physics)
                if obs_flag:
                    obs_list.append(timestep.observation["observations"])
                    action_list.append(action)
                    reward_list.append(self.reward_functions[task](self.env.physics))

            rewards[task] = task_rewards
            if obs_flag:
                observations[task] = np.array(obs_list)[:-1]
                actions[task] = np.array(action_list)
                rewards[task] = np.array(reward_list)

        return rewards, observations, actions
