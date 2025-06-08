# pylint: disable=[W0237, E1123, W0613]
"""Module defining base classed for forward-backward agent."""

import torch

from agents.base import (
    AbstractMLP,
    AbstractActor,
    AbstractGaussianActor,
    AbstractPreprocessor,
)
from typing import Tuple, Optional


class ActorModel(torch.nn.Module):
    """FB/SF agent actor model."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        preprocessed_dimension: int,
        postprocessed_dimension: int,
        z_dimension: int,
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        gaussian_actor: bool,
        actor_activation: torch.nn,
        std_dev_clip: float,
        device: torch.device,
    ):
        super().__init__()

        self.actor = (AbstractGaussianActor if gaussian_actor else AbstractActor)(
            observation_length=postprocessed_dimension * 2,
            action_length=action_length,
            hidden_dimension=actor_hidden_dimension,
            hidden_layers=actor_hidden_layers,
            activation=actor_activation,
            device=device,
            layernorms=False,
        )

        # pre-processors
        self.obs_preprocessor = AbstractPreprocessor(
            observation_length=observation_length,
            concatenated_variable_length=0,  # preprocess observation alone
            hidden_dimension=preprocessed_dimension,
            feature_space_dimension=postprocessed_dimension,
            hidden_layers=1,
            device=device,
            activation="relu",
            layernorm=True,
        )

        self.obs_z_preprocessor = AbstractPreprocessor(
            observation_length=observation_length,
            concatenated_variable_length=z_dimension,
            hidden_dimension=preprocessed_dimension,
            feature_space_dimension=postprocessed_dimension,
            hidden_layers=1,
            device=device,
            activation="relu",
            layernorm=True,
        )

        self._std_dev_clip = std_dev_clip

    def forward(
        self,
        observation: torch.Tensor,
        z: torch.Tensor,
        std: float,
        sample: bool = False,
    ) -> Tuple[torch.Tensor, torch.distributions.Distribution]:
        """
        Takes observation, action, z, preprocesses and forwards into z-space.
        Args:
            observation: state tensor of shape [batch_dim, observation_length]
            z: policy parameter tensor of shape [batch_dim, z_dimension]
            std: standard deviation of the policy
            sample: whether to sample from the policy or not
        Returns:
            action: action tensor of shape [batch_dim, action_length]
        """

        obs_embedding = self.obs_preprocessor(observation)
        obs_z_embedding = self.obs_z_preprocessor(torch.cat([observation, z], dim=-1))
        h = torch.cat([obs_embedding, obs_z_embedding], dim=-1)

        action_dist = (
            self.actor(h)
            if type(self.actor)  # pylint: disable=unidiomatic-typecheck
            == AbstractGaussianActor
            else self.actor(h, std)
        )

        if sample:
            action = (
                action_dist.rsample()
                if type(self.actor)  # pylint: disable=unidiomatic-typecheck
                == AbstractGaussianActor
                else action_dist.sample(clip=self._std_dev_clip)
            )

        else:
            action = action_dist.mean

        return action.clip(-1, 1), action_dist


class ForwardModel(AbstractMLP):
    """
    Predicts the expected future states (measure) given an
    embedding of a current state-action pair and policy parameterised by z.
    """

    def __init__(
        self,
        input_length: int,
        z_dimension: int,
        hidden_dimension: int,
        hidden_layers: int,
        device: torch.device,
        activation: str,
        discrete_actions: bool,
        action_length: int,
    ):
        super().__init__(
            input_dimension=input_length,
            output_dimension=z_dimension,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Takes observation, action, z, preprocesses and forwards into z-space.
        Args:
            h: preprocessed/embedded obs/action/z tensor of shape
             [batch_dim, preprocessor_feature_space_dimension * number_of_features]
        Returns:
            z_: embedded tensor of shape [batch_dim, z_dimension]
        """

        z_ = self.trunk(h)  # pylint: disable=E1102

        return z_


class BackwardModel(AbstractMLP):
    """Backwards model--can be interpreted as the number of ways
    of reaching the observation."""

    def __init__(
        self,
        goal_dimension: int,
        action_dimension: int,
        z_dimension: int,
        hidden_dimension: int,
        hidden_layers: int,
        device: torch.device,
        activation: str,
        include_action: bool,
        discrete_actions: bool,
        layernorm: bool = True,
    ):
        self._include_action = include_action
        if include_action:
            if discrete_actions:
                self.discrete_action_embedding = torch.nn.Embedding(
                    num_embeddings=action_dimension,
                    embedding_dim=action_dimension,
                )
            input_dimension = goal_dimension + action_dimension
        else:
            input_dimension = goal_dimension

        super().__init__(
            input_dimension=input_dimension,
            output_dimension=z_dimension,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
            layernorm=layernorm,
        )
        self._z_dimension = z_dimension

    def forward(
        self, goal: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Takes observation, action, z, preprocesses and forwards into z-space.
        Args:
            goal: goal tensor of shape [batch_dim, goal_dimension]
            action: action tensor of shape
                    [batch_dim, backward_history_length, action_dimension]
        Returns:
            z: embedded tensor of shape [batch_dim, z_dimension]
        """
        if self._include_action:
            action = action.reshape(-1, action.shape[-1])
            if self._discrete_actions:
                action = self.discrete_action_embedding(action)
            h = torch.cat([goal, action], dim=-1)
        else:
            h = goal

        z = self.trunk(h)  # pylint: disable=E1102

        # L2 normalize then scale to radius sqrt(z_dimension)
        z = torch.sqrt(
            torch.tensor(self._z_dimension, dtype=torch.int, device=self.device)
        ) * torch.nn.functional.normalize(z, dim=1)

        return z
