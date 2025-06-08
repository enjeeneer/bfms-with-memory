# pylint: disable=import-outside-toplevel, unused-argument
"""Module for specifying occluded observations on the ExORL benchmarks."""

import abc
import dmc
import numpy as np
import custom_dmc_tasks as cdmc
from utils import get_popgym_env
from utils import EXORL_DOMAINS, POPGYM_DOMAINS


class AbstractOcclusion(metaclass=abc.ABCMeta):
    """
    Abstract class for occluding observations on either Exorl or Popgym.
    """

    def __init__(self, domain: str, obs_type: str, frames: int):

        if domain in EXORL_DOMAINS:
            if domain == "jaco":
                self._base_env = cdmc.make_jaco(
                    task="reach_bottom_left", obs_type="perfect_features", seed=42
                )
                self._flattened_env = dmc.make(name="jaco_reach_bottom_left")
                self._full_observation_length = (
                    self._flattened_env.observation_spec().shape[0]
                )
            else:
                from rewards import DEFAULT_TASKS

                self._base_env = cdmc.make(
                    domain=domain,
                    task=DEFAULT_TASKS[domain],
                    environment_kwargs={"flat_observation": False},
                    obs_type=obs_type,
                    frames=frames,
                )
                self._flattened_env = cdmc.make(
                    domain=domain,
                    task=DEFAULT_TASKS[domain],
                    environment_kwargs={"flat_observation": True},
                    obs_type=obs_type,
                    frames=frames,
                )
                if obs_type == "states":
                    self._full_observation_length = (
                        self._flattened_env.observation_spec()["observations"].shape[0]
                    )
                else:
                    self._full_observation_length = (
                        self._base_env.observation_spec().shape
                    )

        elif domain in POPGYM_DOMAINS:
            env = get_popgym_env(domain=domain)
            if domain in (
                "RepeatPreviousHard",
                "RepeatPreviousMedium",
                "RepeatPreviousEasy",
            ):
                self._full_observation_length = 1
            else:
                self._full_observation_length = env.observation_space.shape[0]

        else:
            raise ValueError(f"{domain} does not have an occlusion implemented.")

    @abc.abstractmethod
    def __call__(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def observation_length(self) -> int:
        raise NotImplementedError


class DefaultOcclusion(AbstractOcclusion):
    """
    Default (i.e. no) occlusion for all envs.
    """

    def __init__(self, domain: str, obs_type: str, frames: int, **kwargs):
        super().__init__(domain=domain, obs_type=obs_type, frames=frames)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        return observation

    @property
    def observation_length(self) -> int:
        return self._full_observation_length


class Flickering(AbstractOcclusion):
    """
    Flickering occlusion for all envs. We have one class for all envs
    because the occlusion is independent of the state space dimension.
    Simulates remote sensor data being lost during long-distance communication.
    """

    def __init__(
        self,
        domain: str,
        frames: int,
        flickering_prob: float,
        obs_type: str = "states",
        **kwargs,
    ):
        super().__init__(domain=domain, obs_type=obs_type, frames=frames)

        if flickering_prob is None:
            raise ValueError("`flickering` must be provided for Flickering POMDPs.")

        self._flickering_prob = flickering_prob

    def __call__(self, observation: np.ndarray) -> np.ndarray:

        init_observation = observation

        # if observation is 1D, reshape to 2D
        if len(init_observation.shape) == 1:
            observation = np.expand_dims(init_observation, 0)

        # create flickered observation
        flickered = np.zeros_like(observation)

        # whether to flicker
        flicker_idxs = (
            np.random.rand(observation.shape[0]) < self._flickering_prob
        ).reshape(
            observation.shape[0],
            *[1] * (len(observation.shape) - 1),
        )

        # add flickering
        observation = np.where(flicker_idxs, flickered, observation)

        # squeeze back to 1D if necessary
        if len(init_observation.shape) == 1:
            # this only happens with np.ndarrays from gym
            observation = np.squeeze(observation)

        return observation

    @property
    def observation_length(self) -> int:
        return self._full_observation_length


class RandomNoise(AbstractOcclusion):
    """
    Random noise occlusion for all envs. We have one class for all envs
    because the occlusion is independent of the state-space dimension.
    Simulates sensor noise.
    """

    def __init__(
        self,
        domain: str,
        frames: int,
        noise_std: float,
        obs_type: str = "states",
        **kwargs,
    ):
        super().__init__(domain=domain, obs_type=obs_type, frames=frames)

        if noise_std is None:
            raise ValueError("`noise_std` must be provided for RandomNoise POMDPs.")

        self._noise_std = noise_std

    def __call__(self, observation: np.ndarray) -> np.ndarray:

        noise = np.random.normal(0, self._noise_std, size=observation.shape).astype(
            np.float32
        )
        return observation + noise

    @property
    def observation_length(self) -> int:
        return self._full_observation_length


class RandomSensorMissing(AbstractOcclusion):
    """
    Random sensor missing occlusion for all envs. We have one class for all envs
    because the occlusion is independent of the state-space dimension.
    Simulates sensors dropping with some probability.
    """

    def __init__(
        self,
        domain: str,
        frames: int,
        missing_sensor_prob: float,
        obs_type: str = "states",
        **kwargs,
    ):
        super().__init__(domain=domain, obs_type=obs_type, frames=frames)

        if missing_sensor_prob is None:
            raise ValueError(
                "`missing_sensor_prob` must be provided for RandomNoise POMDPs."
            )

        self._missing_sensor_prob = missing_sensor_prob

    def __call__(self, observation: np.ndarray) -> np.ndarray:

        init_observation = observation

        # if observation is 1D, reshape to 2D
        if len(init_observation.shape) == 1:
            observation = np.expand_dims(init_observation, 0)

        # which sensors to drop
        missing_idxs = np.random.rand(*observation.shape) < self._missing_sensor_prob

        # drop sensors
        observation[missing_idxs] = 0.0

        # squeeze back to 1D if necessary
        if len(init_observation.shape) == 1:
            # this only happens with np.ndarrays from gym
            observation = np.squeeze(observation)

        return observation

    @property
    def observation_length(self) -> int:
        return self._full_observation_length


class WalkerVelocityOcclusion(AbstractOcclusion):
    """
    Velocity occlusion for point mass maze.
    """

    def __init__(self, domain: str, frames: int, obs_type: str = "states", **kwargs):
        super().__init__(domain=domain, obs_type=obs_type, frames=frames)

        self._orientation_dims = self._base_env.observation_spec()[
            "orientations"
        ].shape[0]
        self._height_dims = 1
        self._velocity_dims = self._base_env.observation_spec()["velocity"].shape[0]

        # masks for non-dynamical dims
        self._orientation_mask = np.arange(self._orientation_dims)
        self._height_mask = np.arange(
            self._orientation_dims, self._orientation_dims + self._height_dims
        )
        self._mask = np.concatenate((self._orientation_mask, self._height_mask))
        self._observation_length = self._orientation_dims + self._height_dims

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        return observation[..., self._mask]

    @property
    def observation_length(self) -> int:
        return self._observation_length


class QuadrupedVelocityOcclusion(AbstractOcclusion):
    """
    Velocity occlusion for point mass maze.
    """

    def __init__(self, domain: str, frames: int, obs_type: str = "states", **kwargs):
        super().__init__(domain=domain, obs_type=obs_type, frames=frames)

        self._egocentric_state_dims = self._base_env.observation_spec()[
            "egocentric_state"
        ].shape[0]
        self._torso_velocity_dims = self._base_env.observation_spec()[
            "torso_velocity"
        ].shape[0]
        self._torso_upright_dims = 1
        self._inertial_measurement_unit_dims = self._base_env.observation_spec()[
            "imu"
        ].shape[0]
        self._torso_torque_dims = self._base_env.observation_spec()[
            "force_torque"
        ].shape[0]

        # masks for non-dynamical dims
        self._egocentric_state_mask = np.arange(self._egocentric_state_dims)
        self._torso_upright_mask = np.arange(
            self._egocentric_state_dims + self._torso_velocity_dims,
            self._egocentric_state_dims
            + self._torso_velocity_dims
            + self._torso_upright_dims,
        )
        self._force_torque_mask = np.arange(
            self._egocentric_state_dims
            + self._torso_velocity_dims
            + self._torso_upright_dims
            + self._inertial_measurement_unit_dims,
            self._egocentric_state_dims
            + self._torso_velocity_dims
            + self._torso_upright_dims
            + self._inertial_measurement_unit_dims
            + self._torso_torque_dims,
        )
        self._mask = np.concatenate(
            (
                self._egocentric_state_mask,
                self._torso_upright_mask,
                self._force_torque_mask,
            )
        )
        self._observation_length = (
            self._egocentric_state_dims
            + self._torso_upright_dims
            + self._torso_torque_dims
        )

    def __call__(self, domain: str, observation: np.ndarray) -> np.ndarray:
        return observation[..., self._mask]

    @property
    def observation_length(self) -> int:
        return self._observation_length


class CheetahVelocityOcclusion(AbstractOcclusion):
    """
    Velocity occlusion for point mass maze.
    """

    def __init__(self, domain: str, frames: int, obs_type: str = "states", **kwargs):
        super().__init__(domain=domain, obs_type=obs_type, frames=frames)

        self._position_dims = self._base_env.observation_spec()["position"].shape[0]
        self._velocity_dims = self._base_env.observation_spec()["velocity"].shape[0]

        # masks for non-dynamical dims
        self._position_mask = np.arange(self._position_dims)
        self._mask = self._position_mask
        self._observation_length = self._position_dims

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        return observation[..., self._mask]

    @property
    def observation_length(self) -> int:
        return self._observation_lengthDefaultOcclusion


# seperate dictionaries for dynamics and goal occlusions
DYNAMICS_OCCLUSIONS = {
    # --- walker ---
    "walker": DefaultOcclusion,
    "walker_dynamics_occluded": WalkerVelocityOcclusion,
    "walker_dynamics_flickering": Flickering,
    "walker_dynamics_noise": RandomNoise,
    "walker_dynamics_sensors": RandomSensorMissing,
    "walker_rewards_occluded": DefaultOcclusion,
    "walker_rewards_flickering": DefaultOcclusion,
    "walker_rewards_noise": DefaultOcclusion,
    "walker_rewards_sensors": DefaultOcclusion,
    "walker_occluded": WalkerVelocityOcclusion,
    "walker_flickering": Flickering,
    "walker_noise": RandomNoise,
    "walker_sensors": RandomSensorMissing,
    # --- quadruped ---
    "quadruped": DefaultOcclusion,
    "quadruped_dynamics_occluded": QuadrupedVelocityOcclusion,
    "quadruped_dynamics_flickering": Flickering,
    "quadruped_dynamics_noise": RandomNoise,
    "quadruped_dynamics_sensors": RandomSensorMissing,
    "quadruped_rewards_occluded": DefaultOcclusion,
    "quadruped_rewards_flickering": DefaultOcclusion,
    "quadruped_rewards_noise": DefaultOcclusion,
    "quadruped_rewards_sensors": DefaultOcclusion,
    "quadruped_occluded": QuadrupedVelocityOcclusion,
    "quadruped_flickering": Flickering,
    "quadruped_noise": RandomNoise,
    "quadruped_sensors": RandomSensorMissing,
    # --- cheetah ---
    "cheetah": DefaultOcclusion,
    "cheetah_dynamics_occluded": CheetahVelocityOcclusion,
    "cheetah_dynamics_flickering": Flickering,
    "cheetah_dynamics_noise": RandomNoise,
    "cheetah_dynamics_sensors": RandomSensorMissing,
    "cheetah_rewards_occluded": DefaultOcclusion,
    "cheetah_rewards_flickering": DefaultOcclusion,
    "cheetah_rewards_noise": DefaultOcclusion,
    "cheetah_rewards_sensors": DefaultOcclusion,
    "cheetah_occluded": CheetahVelocityOcclusion,
    "cheetah_flickering": Flickering,
    "cheetah_noise": RandomNoise,
    "cheetah_sensors": RandomSensorMissing,
    # --- popgym ---
    "RepeatPreviousHard": DefaultOcclusion,
    "RepeatPreviousMedium": DefaultOcclusion,
    "RepeatPreviousEasy": DefaultOcclusion,
    "NoisyStatelessPendulumHard": DefaultOcclusion,
    "StatelessPendulumHard": DefaultOcclusion,
    "NoisyStatelessCartPoleHard": DefaultOcclusion,
    "StatelessCartPoleHard": DefaultOcclusion,
}

REWARD_OCCLUSIONS = {
    # --- walker ---
    "walker": DefaultOcclusion,
    "walker_dynamics_occluded": DefaultOcclusion,
    "walker_dynamics_flickering": DefaultOcclusion,
    "walker_dynamics_noise": DefaultOcclusion,
    "walker_dynamics_sensors": DefaultOcclusion,
    "walker_rewards_occluded": WalkerVelocityOcclusion,
    "walker_rewards_flickering": Flickering,
    "walker_rewards_noise": RandomNoise,
    "walker_rewards_sensors": RandomSensorMissing,
    "walker_occluded": WalkerVelocityOcclusion,
    "walker_flickering": Flickering,
    "walker_noise": RandomNoise,
    "walker_sensors": RandomSensorMissing,
    # --- quadruped ---
    "quadruped": DefaultOcclusion,
    "quadruped_dynamics_occluded": DefaultOcclusion,
    "quadruped_dynamics_flickering": DefaultOcclusion,
    "quadruped_dynamics_noise": DefaultOcclusion,
    "quadruped_dynamics_sensors": DefaultOcclusion,
    "quadruped_rewards_occluded": QuadrupedVelocityOcclusion,
    "quadruped_rewards_flickering": Flickering,
    "quadruped_rewards_noise": RandomNoise,
    "quadruped_rewards_sensors": RandomSensorMissing,
    "quadruped_occluded": QuadrupedVelocityOcclusion,
    "quadruped_flickering": Flickering,
    "quadruped_noise": RandomNoise,
    "quadruped_sensors": RandomSensorMissing,
    # --- cheetah ---
    "cheetah": DefaultOcclusion,
    "cheetah_dynamics_occluded": DefaultOcclusion,
    "cheetah_dynamics_flickering": DefaultOcclusion,
    "cheetah_dynamics_noise": DefaultOcclusion,
    "cheetah_dynamics_sensors": DefaultOcclusion,
    "cheetah_rewards_occluded": CheetahVelocityOcclusion,
    "cheetah_rewards_flickering": Flickering,
    "cheetah_rewards_noise": RandomNoise,
    "cheetah_rewards_sensors": RandomSensorMissing,
    "cheetah_occluded": CheetahVelocityOcclusion,
    "cheetah_flickering": Flickering,
    "cheetah_noise": RandomNoise,
    "cheetah_sensors": RandomSensorMissing,
    # --- popgym ---
    "RepeatPreviousHard": DefaultOcclusion,
    "RepeatPreviousMedium": DefaultOcclusion,
    "RepeatPreviousEasy": DefaultOcclusion,
    "NoisyStatelessPendulumHard": DefaultOcclusion,
    "StatelessPendulumHard": DefaultOcclusion,
    "NoisyStatelessCartPoleHard": DefaultOcclusion,
    "StatelessCartPoleHard": DefaultOcclusion,
}
