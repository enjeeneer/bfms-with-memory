"""RND module."""

import torch
import numpy as np
from typing import Dict, Any

from agents.td3.agent import TD3
from agents.utils import RunningMeanStd
from agents.base import Batch


class RNDModels(torch.torch.nn.Module):
    """
    Random Network Distillation module.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        rnd_rep_dim: int,
        obs_shape: np.array,
        clip_val=5.0,
    ) -> None:

        super().__init__()

        self.clip_val = clip_val
        self.normalize_obs = torch.nn.BatchNorm1d(obs_shape[0], affine=False)

        # networks
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, rnd_rep_dim),
        )
        self.target = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, rnd_rep_dim),
        )

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, observation) -> torch.Tensor:
        """
        Forward pass of the RND module.
        Args:
            observation: tensor of shape [batch_dim, obs_dim]
        Returns:
            prediction_error: tensor of shape [batch_dim, 1]
        """

        observation = self.normalize_obs(observation)
        observation = torch.clamp(observation, -self.clip_val, self.clip_val)

        # get prediction errors
        prediction, target = self.predictor(observation), self.target(observation)
        prediction_error = torch.square(target.detach() - prediction).mean(
            dim=-1, keepdim=True
        )

        return prediction_error


class RND(TD3):
    """RND agent built atop TD3."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        device: torch.device,
        critic_hidden_dimension: int,
        critic_hidden_layers: int,
        critic_learning_rate: float,
        critic_target_update_fequency: int,
        actor_update_frequency: int,
        critic_activation: str,
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        actor_learning_rate: float,
        actor_activation: str,
        std_dev_clip: float,
        std_dev_schedule: str,
        batch_size: int,
        discount: float,
        tau: float,
        rnd_representation_dimension: int,
        rnd_hidden_dimension: int,
        rnd_scale: float,
        discrete_actions: bool = False,
        non_episodic: bool = True,
    ):
        super().__init__(
            observation_length,
            action_length,
            device,
            "rnd",
            critic_hidden_dimension,
            critic_hidden_layers,
            critic_learning_rate,
            critic_activation,
            critic_target_update_fequency,
            actor_update_frequency,
            actor_hidden_dimension,
            actor_hidden_layers,
            actor_learning_rate,
            actor_activation,
            std_dev_clip,
            std_dev_schedule,
            batch_size,
            discount,
            tau,
            discrete_actions=discrete_actions,
        )

        self.rnd = RNDModels(
            obs_dim=observation_length,
            hidden_dim=rnd_hidden_dimension,
            rnd_rep_dim=rnd_representation_dimension,
            obs_shape=(observation_length,),
        ).to(self.device)

        self.intrinsic_reward_rms = RunningMeanStd(device=self.device)
        self.rnd_scale = rnd_scale
        self.discrete_actions = discrete_actions
        self.non_episodic = non_episodic

        # optimizers
        self.rnd_opt = torch.optim.Adam(self.rnd.parameters(), lr=critic_learning_rate)

        self.rnd.train()

    def update_rnd(self, observations: torch.Tensor) -> Dict[str, Any]:
        """
        Update the RND module.
        Args:
            observations: tensor of shape [batch_dim, obs_dim]
        Returns:
            metrics: dictionary of metrics
        """
        metrics = {}
        prediction_error = self.rnd.forward(observation=observations)

        loss = prediction_error.mean()

        self.rnd_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.rnd_opt.step()

        metrics["train/rnd_loss"] = loss.item()

        return metrics

    def compute_intrinsic_reward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Compute the RND intrinsic reward.
        Args:
            observation: tensor of shape [batch_dim, obs_dim]
        Returns:
            reward: tensor of shape [batch_dim, 1]
        """

        prediction_error = self.rnd.forward(observation=observation)
        _, intr_reward_var = self.intrinsic_reward_rms(prediction_error)
        reward = (
            self.rnd_scale * prediction_error / (torch.sqrt(intr_reward_var) + 1e-8)
        )

        return reward

    def update(self, batch: Batch, step: int) -> Dict[str, float]:
        """
        Updates agent's networks given a batch_size sample from the replay buffer.
        Args:
            batch: memory buffer containing transitions
            step: no. of steps taken in the environment
        Returns:
            metrics: dictionary of metrics for logging
        """

        (observations, actions, next_observations, not_dones) = (
            batch.observations,
            batch.actions,
            batch.next_observations,
            batch.not_dones,
        )

        if self.non_episodic:
            # hardcode all not dones to make agent non-episodic
            not_dones = torch.ones_like(not_dones)

        # update RND first
        rnd_metrics = self.update_rnd(observations=observations)

        with torch.no_grad():
            intrinsic_rewards = self.compute_intrinsic_reward(observations)

        critic_metrics = self._update_critic(
            observations=observations,
            actions=actions,
            rewards=intrinsic_rewards,
            next_observations=next_observations,
            not_dones=not_dones,
            step=step,
        )

        actor_metrics = {}
        if not self.discrete_actions:
            if step % self.actor_update_frequency == 0:
                actor_metrics = self._update_actor(observations=observations, step=step)

        # polyak critic target update
        if step % self.critic_target_update_frequency == 0:
            self._soft_critic_target_update()

        # logging
        metrics = {
            **actor_metrics,
            **critic_metrics,
            **rnd_metrics,
            "train/pred_error_mean": self.intrinsic_reward_rms.M.item(),
            "train/pred_error_std": torch.sqrt(self.intrinsic_reward_rms.S).item(),
            "train/intrinsic_reward": intrinsic_rewards.mean().item(),
        }

        return metrics
