# pylint: disable=unused-argument
"""Module for fully defined models used by the forward-backward agent. """

from agents.fb.base import (
    ActorModel,
    ForwardModel,
    BackwardModel,
)
from agents.base import AbstractPreprocessor
from typing import Tuple, Optional

import torch


class ForwardRepresentation(torch.nn.Module):
    """Forward representation network."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        preprocessed_dimension: int,
        postprocessed_dimension: int,
        z_dimension: int,
        forward_hidden_dimension: int,
        forward_hidden_layers: int,
        device: torch.device,
        forward_activation: str,
        discrete_actions: bool,
        layernorms: bool,
    ):
        super().__init__()

        # pre-processors
        self.obs_action_preprocessor = AbstractPreprocessor(
            observation_length=observation_length,
            concatenated_variable_length=0 if discrete_actions else action_length,
            hidden_dimension=preprocessed_dimension,
            feature_space_dimension=postprocessed_dimension,
            hidden_layers=1,
            device=device,
            activation="relu",
            layernorm=layernorms,
        )

        self.obs_z_preprocessor = AbstractPreprocessor(
            observation_length=observation_length,
            concatenated_variable_length=z_dimension,
            hidden_dimension=preprocessed_dimension,
            feature_space_dimension=postprocessed_dimension,
            hidden_layers=1,
            device=device,
            activation="relu",
            layernorm=layernorms,
        )

        forward_input_length = postprocessed_dimension * 2

        self.F1 = ForwardModel(
            input_length=forward_input_length,
            z_dimension=z_dimension * action_length
            if discrete_actions
            else z_dimension,
            hidden_dimension=forward_hidden_dimension,
            hidden_layers=forward_hidden_layers,
            device=device,
            activation=forward_activation,
            discrete_actions=discrete_actions,
            action_length=action_length,
        )

        self.F2 = ForwardModel(
            input_length=forward_input_length,
            z_dimension=z_dimension * action_length
            if discrete_actions
            else z_dimension,
            hidden_dimension=forward_hidden_dimension,
            hidden_layers=forward_hidden_layers,
            device=device,
            activation=forward_activation,
            discrete_actions=discrete_actions,
            action_length=action_length,
        )

        self._discrete_actions = discrete_actions
        self._z_dimension = z_dimension
        self._action_length = action_length

    def forward(
        self,
        observation: torch.Tensor,
        z: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passes observations, actions, and zs through both forward models.
        """
        if self._discrete_actions:
            obs_embedding = self.obs_action_preprocessor(observation)
        else:
            obs_embedding = self.obs_action_preprocessor(
                torch.cat([observation, action], dim=-1)
            )
        obs_z_embedding = self.obs_z_preprocessor(torch.cat([observation, z], dim=-1))

        h = torch.cat([obs_embedding, obs_z_embedding], dim=-1)

        F1, F2 = self.F1(h), self.F2(h)

        if self._discrete_actions:
            F1 = F1.reshape(-1, self._z_dimension, self._action_length)
            F2 = F2.reshape(-1, self._z_dimension, self._action_length)

        return F1, F2


class BackwardRepresentation(torch.nn.Module):
    """Backward representation network."""

    def __init__(
        self,
        goal_dimension: int,
        action_dimension: int,
        z_dimension: int,
        backward_hidden_dimension: int,
        backward_hidden_layers: int,
        device: torch.device,
        backward_activation: torch.nn,
        layernorms: bool,
        include_action: bool,
        discrete_actions: bool,
    ):
        super().__init__()

        self.B = BackwardModel(
            goal_dimension=goal_dimension,
            action_dimension=action_dimension,
            z_dimension=z_dimension,
            hidden_dimension=backward_hidden_dimension,
            hidden_layers=backward_hidden_layers,
            device=device,
            activation=backward_activation,
            layernorm=layernorms,
            include_action=include_action,
            discrete_actions=discrete_actions,
        )

    def forward(
        self,
        goal: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Estimates routes to observation via backwards model.
        Args:
            goal: tensor of shape [batch_dim, goal_dimension]
            action: tensor of shape [batch_dim, action_dimension]
        Returns:
            z: tensor of shape [batch_dim, z_dimension]
        """

        return self.B(goal=goal, action=action)


class ForwardBackwardRepresentation(torch.nn.Module):
    """Combined Forward-backward representation network."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        goal_dimension: int,
        preprocessed_dimension: int,
        postprocessed_dimension: int,
        z_dimension: int,
        forward_hidden_dimension: int,
        forward_hidden_layers: int,
        backward_hidden_dimension: int,
        backward_hidden_layers: int,
        forward_activation: str,
        backward_activation: str,
        orthonormalisation_coefficient: float,
        discount: float,
        device: torch.device,
        discrete_actions: bool,
        layernorms: bool,
        include_action_in_B: bool,
    ):
        super().__init__()
        self.forward_representation = ForwardRepresentation(
            observation_length=observation_length,
            action_length=action_length,
            preprocessed_dimension=preprocessed_dimension,
            postprocessed_dimension=postprocessed_dimension,
            z_dimension=z_dimension,
            forward_hidden_dimension=forward_hidden_dimension,
            forward_hidden_layers=forward_hidden_layers,
            device=device,
            forward_activation=forward_activation,
            discrete_actions=discrete_actions,
            layernorms=layernorms,
        )

        self.backward_representation = BackwardRepresentation(
            goal_dimension=goal_dimension,
            action_dimension=action_length,
            z_dimension=z_dimension,
            backward_hidden_dimension=backward_hidden_dimension,
            backward_hidden_layers=backward_hidden_layers,
            device=device,
            backward_activation=backward_activation,
            layernorms=layernorms,
            include_action=include_action_in_B,
            discrete_actions=discrete_actions,
        )

        self.forward_representation_target = ForwardRepresentation(
            observation_length=observation_length,
            action_length=action_length,
            preprocessed_dimension=preprocessed_dimension,
            postprocessed_dimension=postprocessed_dimension,
            z_dimension=z_dimension,
            forward_hidden_dimension=forward_hidden_dimension,
            forward_hidden_layers=forward_hidden_layers,
            device=device,
            forward_activation=forward_activation,
            discrete_actions=discrete_actions,
            layernorms=layernorms,
        )

        self.backward_representation_target = BackwardRepresentation(
            goal_dimension=goal_dimension,
            action_dimension=action_length,
            z_dimension=z_dimension,
            backward_hidden_dimension=backward_hidden_dimension,
            backward_hidden_layers=backward_hidden_layers,
            device=device,
            backward_activation=backward_activation,
            layernorms=layernorms,
            include_action=include_action_in_B,
            discrete_actions=discrete_actions,
        )

        self._discount = discount
        self.orthonormalisation_coefficient = orthonormalisation_coefficient
        self._device = device


class Actor(torch.nn.Module):
    """Action selecting network."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        preprocessed_dimension: int,
        postprocessed_dimension: int,
        z_dimension: int,
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        actor_activation: torch.nn,
        std_dev_schedule: str,
        std_dev_clip: float,
        device: torch.device,
        gaussian_actor: bool,
    ):
        super().__init__()

        self.actor = ActorModel(
            observation_length=observation_length,
            action_length=action_length,
            preprocessed_dimension=preprocessed_dimension,
            postprocessed_dimension=postprocessed_dimension,
            z_dimension=z_dimension,
            actor_hidden_dimension=actor_hidden_dimension,
            actor_hidden_layers=actor_hidden_layers,
            actor_activation=actor_activation,
            std_dev_clip=std_dev_clip,
            device=device,
            gaussian_actor=gaussian_actor,
        )

        self._std_dev_schedule = std_dev_schedule

    def forward(
        self,
        observation: torch.Tensor,
        z: torch.Tensor,
        std: float,
        beta: Optional[torch.Tensor] = None,
        sample: bool = False,
    ) -> Tuple[torch.Tensor, torch.distributions.Distribution]:
        """
        Returns actions from actor model.
        Args:
            observation: observation tensor of shape [batch_dim, observation_length]
            z: task tensor of shape [batch_dim, z_dimension]
            std: standard deviation for action distribution
            beta: (optional) SF augmentation of shape [batch_dim, observation_length]
                    only passed for FB-delta variant
            sample: whether to sample from action distribution
        """
        action, action_dist = self.actor(observation, z, std, beta, sample)

        return action, action_dist
