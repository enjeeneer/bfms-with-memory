# pylint: disable=unused-variable
"""Module for fully defined models used by the forward-backward agent. """

from agents.base import (
    LSTMEncoder,
    GRUEncoder,
    S4DEncoder,
    TransformerEncoder,
    AbstractPreprocessor,
    MLPEncoder,
)
from agents.fb_m.base import ForwardModel, BackwardModel, ActorModel
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


class RecurrentForwardRepresentation(torch.nn.Module):
    """
    Recurrent forward representation network.
    """

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        preprocessed_dimension: int,
        postprocessed_dimension: int,
        memory_type: str,
        history_length: int,
        z_dimension: int,
        forward_hidden_dimension: int,
        forward_hidden_layers: int,
        device: torch.device,
        forward_activation: str,
        num_encoder_layers: int,
        discrete_actions: bool,
        discrete_embedding_dimension: int = 16,
        pooling: Optional[str] = None,
        transformer_dimension: Optional[int] = None,
        gru_dimension: Optional[int] = None,
        transformer_attention: Optional[str] = None,
        s4_dimension: Optional[int] = None,
        transformer_n_heads: Optional[int] = None,
    ):
        super().__init__()

        self._discrete_actions = discrete_actions
        self.device = device
        # embedding network for discrete actions
        if self._discrete_actions:
            self.discrete_action_embedding = torch.nn.Embedding(
                num_embeddings=4, embedding_dim=discrete_embedding_dimension
            ).to(device)
            self.discrete_state_embedding = torch.nn.Embedding(
                num_embeddings=4, embedding_dim=discrete_embedding_dimension
            ).to(device)
            obs_action_input_dimension = 2 * discrete_embedding_dimension

        else:
            # pre-processors
            if memory_type == "mlp":
                # frame-stacking
                obs_action_input_dimension = (
                    observation_length + action_length
                ) * history_length
            else:
                obs_action_input_dimension = observation_length + action_length

        if memory_type == "lstm":
            self.obs_action_preprocessor = LSTMEncoder(
                raw_input_dimension=obs_action_input_dimension,
                preprocessed_dimension=preprocessed_dimension,
                postprocessed_dimension=postprocessed_dimension,
                device=device,
            )
            self.obs_z_preprocessor = AbstractPreprocessor(
                observation_length=discrete_embedding_dimension
                if discrete_actions
                else observation_length,
                concatenated_variable_length=z_dimension,
                hidden_dimension=preprocessed_dimension,
                feature_space_dimension=postprocessed_dimension,
                hidden_layers=1,
                device=device,
                activation="relu",
                layernorm=True,
            )

        elif memory_type == "transformer":
            self.obs_action_preprocessor = TransformerEncoder(
                raw_input_dimension=obs_action_input_dimension,
                preprocessed_dimension=preprocessed_dimension,
                transformer_dimension=transformer_dimension,
                transformer_n_heads=transformer_n_heads,
                num_layers=num_encoder_layers,
                postprocessed_dimension=postprocessed_dimension,
                transformer_attention=transformer_attention,
                device=device,
                pooling=pooling,
                history_length=history_length,
            )
            self.obs_z_preprocessor = AbstractPreprocessor(
                observation_length=discrete_embedding_dimension
                if discrete_actions
                else observation_length,
                concatenated_variable_length=z_dimension,
                hidden_dimension=preprocessed_dimension,
                feature_space_dimension=postprocessed_dimension,
                hidden_layers=1,
                device=device,
                activation="relu",
                layernorm=True,
            )

        elif memory_type == "s4d":
            self.obs_action_preprocessor = S4DEncoder(
                raw_input_dimension=obs_action_input_dimension,
                preprocessed_dimension=preprocessed_dimension,
                postprocessed_dimension=postprocessed_dimension,
                s4_dimension=s4_dimension,
                device=device,
                num_layers=num_encoder_layers,
            )
            self.obs_z_preprocessor = AbstractPreprocessor(
                observation_length=discrete_embedding_dimension
                if discrete_actions
                else observation_length,
                concatenated_variable_length=z_dimension,
                hidden_dimension=preprocessed_dimension,
                feature_space_dimension=postprocessed_dimension,
                hidden_layers=1,
                device=device,
                activation="relu",
                layernorm=True,
            )

        elif memory_type == "gru":
            self.obs_action_preprocessor = GRUEncoder(
                raw_input_dimension=obs_action_input_dimension,
                preprocessed_dimension=preprocessed_dimension,
                postprocessed_dimension=postprocessed_dimension,
                device=device,
                num_layers=num_encoder_layers,
                gru_dimension=gru_dimension,
            )
            self.obs_z_preprocessor = AbstractPreprocessor(
                observation_length=discrete_embedding_dimension
                if discrete_actions
                else observation_length,
                concatenated_variable_length=z_dimension,
                hidden_dimension=preprocessed_dimension,
                feature_space_dimension=postprocessed_dimension,
                hidden_layers=1,
                device=device,
                activation="relu",
                layernorm=True,
            )

        elif memory_type == "mlp":
            self.obs_action_preprocessor = MLPEncoder(
                raw_input_dimension=obs_action_input_dimension,
                preprocessed_dimension=preprocessed_dimension,
                postprocessed_dimension=postprocessed_dimension,
                device=device,
                layers=2,
            )
            self.obs_z_preprocessor = AbstractPreprocessor(
                observation_length=discrete_embedding_dimension
                if discrete_actions
                else observation_length,
                concatenated_variable_length=z_dimension,
                hidden_dimension=preprocessed_dimension,
                feature_space_dimension=postprocessed_dimension,
                hidden_layers=1,
                device=device,
                activation="relu",
                layernorm=True,
            )
        else:
            raise ValueError(f"Invalid memory type: {memory_type}")

        forward_model_input_length = 2 * postprocessed_dimension

        self.F1 = ForwardModel(
            input_length=forward_model_input_length,
            z_dimension=z_dimension,
            hidden_dimension=forward_hidden_dimension,
            hidden_layers=forward_hidden_layers,
            device=device,
            activation=forward_activation,
            discrete_actions=discrete_actions,
            action_length=action_length,
        )

        self.F2 = ForwardModel(
            input_length=forward_model_input_length,
            z_dimension=z_dimension,
            hidden_dimension=forward_hidden_dimension,
            hidden_layers=forward_hidden_layers,
            device=device,
            activation=forward_activation,
            discrete_actions=discrete_actions,
            action_length=action_length,
        )

        self._z_dimension = z_dimension
        self._action_length = action_length

    def init_internal_state(
        self,
    ) -> Tuple[torch.Tensor, None]:
        """
        Returns initial internal states of GRU encoder(s).
        """
        return self.obs_action_preprocessor.init_internal_state(), None

    def forward(
        self,
        observation_history: torch.Tensor,
        z_history: torch.Tensor,
        first_time_idx: torch.Tensor,
        action_history: Optional[torch.Tensor] = None,
        prev_hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Passes observations, actions, and zs through both forward models.
        Args:
            observation_history: context tensor of shape [batch_dim,
                                history_length, observation_length]
            first_time_idx: time index tensor which indicates the index of
                        the first observation in the history;
                            shape [batch_dim, 1]
            z_history: context tensor of shape [batch_dim, history_length, z_dimension]
            action_history: context tensor of shape [batch_dim,
                                                        history_length, action_length]
            prev_hidden_state: hidden state from previous forward pass of the forward
                            model. Only used for discrete action spaces
        Returns:
            F1: output of first forward model
            F2: output of second forward model
            hidden_state: final hidden state of memory module
        """
        if self._discrete_actions:
            # here the sequence is unrolled for all discrete actions, then indexed
            action_history = self.discrete_action_embedding.forward(
                action_history.to(device=self.device, dtype=torch.int)
            ).squeeze(-2)
            observation_history = self.discrete_state_embedding.forward(
                observation_history.to(device=self.device, dtype=torch.int)
            ).squeeze(-2)

        obs_action_embedding, hidden_state = self.obs_action_preprocessor.forward(
            history=torch.cat([observation_history, action_history], dim=-1),
            first_time_idx=first_time_idx,
            previous_hidden_state=prev_hidden_state,
        )

        obs_z_embedding = self.obs_z_preprocessor(
            torch.cat([observation_history[:, -1, :], z_history[:, -1, :]], dim=-1)
        )

        h = torch.cat([obs_action_embedding, obs_z_embedding], dim=-1)

        F1, F2 = self.F1(h), self.F2(h)

        # if self._discrete_actions:
        #     F1 = F1.reshape(-1, self._z_dimension, self._action_length)
        #     F2 = F2.reshape(-1, self._z_dimension, self._action_length)

        return F1, F2, hidden_state


class RecurrentBackwardRepresentation(torch.nn.Module):
    """Backward representation network."""

    def __init__(
        self,
        goal_dimension: int,
        action_length: int,
        z_dimension: int,
        backward_hidden_dimension: int,
        device: torch.device,
        memory_type: str,
        history_length: int,
        num_encoder_layers: int,
        discrete_actions: bool,
        pooling: Optional[str] = None,
        discrete_embedding_dimension: int = 16,
        transformer_dimension: Optional[int] = None,
        gru_dimension: Optional[int] = None,
        transformer_attention: Optional[str] = None,
        s4_dimension: Optional[int] = None,
        transformer_n_heads: Optional[int] = None,
    ):
        super().__init__()

        self._discrete_actions = discrete_actions
        # embedding network for discrete actions
        if self._discrete_actions:
            self.discrete_action_embedding = torch.nn.Embedding(
                num_embeddings=4, embedding_dim=discrete_embedding_dimension
            ).to(device)
            self.discrete_state_embedding = torch.nn.Embedding(
                num_embeddings=4, embedding_dim=discrete_embedding_dimension
            ).to(device)
            goal_action_input_dimension = 2 * discrete_embedding_dimension
        else:
            # pre-processors
            if memory_type == "mlp":
                # frame-stacking
                goal_action_input_dimension = (
                    goal_dimension + action_length
                ) * history_length
            else:
                goal_action_input_dimension = goal_dimension + action_length

        if memory_type == "transformer":
            self.B = TransformerEncoder(
                raw_input_dimension=goal_action_input_dimension,
                preprocessed_dimension=backward_hidden_dimension,
                transformer_dimension=transformer_dimension,
                transformer_n_heads=transformer_n_heads,
                num_layers=num_encoder_layers,
                postprocessed_dimension=z_dimension,
                transformer_attention=transformer_attention,
                device=device,
                pooling=pooling,
                history_length=history_length,
            )

        elif memory_type == "s4d":
            self.B = S4DEncoder(
                raw_input_dimension=goal_action_input_dimension,
                preprocessed_dimension=backward_hidden_dimension,
                postprocessed_dimension=z_dimension,
                s4_dimension=s4_dimension,
                device=device,
                num_layers=num_encoder_layers,
            )

        elif memory_type == "gru":
            self.B = GRUEncoder(
                raw_input_dimension=goal_action_input_dimension,
                preprocessed_dimension=backward_hidden_dimension,
                postprocessed_dimension=z_dimension,
                device=device,
                num_layers=num_encoder_layers,
                gru_dimension=gru_dimension,
            )

        elif memory_type == "mlp":
            self.B = MLPEncoder(
                raw_input_dimension=goal_action_input_dimension,
                preprocessed_dimension=backward_hidden_dimension,
                postprocessed_dimension=z_dimension,
                device=device,
                layers=num_encoder_layers,
            )

        else:
            raise ValueError(f"Invalid memory type: {memory_type}")

        self._z_dimension = z_dimension
        self.device = device

    def forward(
        self,
        goal: torch.Tensor,
        action: torch.Tensor,
        first_time_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimates routes to observation via backwards model.
        Args:
            goal: context tensor of shape [batch_dim,
                                history_length, goal_dimension]
            action: context tensor of shape [batch_dim,
                                history_length, action_length]
            first_time_idx: time index tensor which indicates the index of
                        the first observation in the history;
                            shape [batch_dim, 1]
        Returns:
            z: tensor of shape [batch_dim, z_dimension]
        """
        if self._discrete_actions:
            goal = self.discrete_state_embedding.forward(
                goal.to(device=self.device, dtype=torch.int)
            ).squeeze(-2)
            action = self.discrete_action_embedding.forward(
                action.to(device=self.device, dtype=torch.int)
            ).squeeze(-2)

        goal_action_embedding, _ = self.B(
            torch.concat([goal, action], dim=-1),
            first_time_idx=first_time_idx,
        )

        # L2 normalize then scale to radius sqrt(z_dimension)
        z = torch.sqrt(
            torch.tensor(self._z_dimension, dtype=torch.int, device=self.device)
        ) * torch.nn.functional.normalize(goal_action_embedding, dim=1)

        return z


class RecurrentForwardBackwardRepresentation(torch.nn.Module):
    """Combined recurrent Forward-backward representation network."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        goal_dimension: int,
        goal_frames: int,
        preprocessed_dimension: int,
        postprocessed_dimension: int,
        memory_type: str,
        history_length: int,
        backward_history_length: int,
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
        layernorms: bool,
        num_encoder_layers: int,
        discrete_actions: bool,
        recurrent_F: bool,
        recurrent_B: bool,
        pooling: Optional[str] = None,
        transformer_dimension: Optional[int] = None,
        gru_dimension: Optional[int] = None,
        transformer_attention: Optional[str] = None,
        s4_dimension: Optional[int] = None,
        transformer_n_heads: Optional[int] = None,
    ):
        super().__init__()
        if recurrent_F:
            self.forward_representation = RecurrentForwardRepresentation(
                observation_length=observation_length,
                action_length=action_length,
                preprocessed_dimension=preprocessed_dimension,
                postprocessed_dimension=postprocessed_dimension,
                memory_type=memory_type,
                history_length=history_length,
                z_dimension=z_dimension,
                forward_hidden_dimension=forward_hidden_dimension,
                forward_hidden_layers=forward_hidden_layers,
                device=device,
                forward_activation=forward_activation,
                pooling=pooling,
                transformer_dimension=transformer_dimension,
                transformer_attention=transformer_attention,
                s4_dimension=s4_dimension,
                transformer_n_heads=transformer_n_heads,
                num_encoder_layers=num_encoder_layers,
                discrete_actions=discrete_actions,
                gru_dimension=gru_dimension,
            )
            self.forward_representation_target = RecurrentForwardRepresentation(
                observation_length=observation_length,
                action_length=action_length,
                preprocessed_dimension=preprocessed_dimension,
                postprocessed_dimension=postprocessed_dimension,
                memory_type=memory_type,
                history_length=history_length,
                z_dimension=z_dimension,
                forward_hidden_dimension=forward_hidden_dimension,
                transformer_attention=transformer_attention,
                forward_hidden_layers=forward_hidden_layers,
                device=device,
                forward_activation=forward_activation,
                pooling=pooling,
                transformer_dimension=transformer_dimension,
                s4_dimension=s4_dimension,
                transformer_n_heads=transformer_n_heads,
                num_encoder_layers=num_encoder_layers,
                discrete_actions=discrete_actions,
                gru_dimension=gru_dimension,
            )
        else:
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

        if recurrent_B:
            self.backward_representation = RecurrentBackwardRepresentation(
                goal_dimension=goal_dimension,
                z_dimension=z_dimension,
                backward_hidden_dimension=backward_hidden_dimension,
                device=device,
                memory_type=memory_type,
                history_length=backward_history_length,
                num_encoder_layers=num_encoder_layers,
                pooling=pooling,
                transformer_dimension=transformer_dimension,
                transformer_attention=transformer_attention,
                s4_dimension=s4_dimension,
                transformer_n_heads=transformer_n_heads,
                gru_dimension=gru_dimension,
                action_length=action_length,
                discrete_actions=discrete_actions,
            )
            self.backward_representation_target = RecurrentBackwardRepresentation(
                goal_dimension=goal_dimension,
                z_dimension=z_dimension,
                backward_hidden_dimension=backward_hidden_dimension,
                device=device,
                memory_type=memory_type,
                history_length=backward_history_length,
                num_encoder_layers=num_encoder_layers,
                pooling=pooling,
                transformer_dimension=transformer_dimension,
                transformer_attention=transformer_attention,
                s4_dimension=s4_dimension,
                transformer_n_heads=transformer_n_heads,
                gru_dimension=gru_dimension,
                action_length=action_length,
                discrete_actions=discrete_actions,
            )

        else:
            self.backward_representation = BackwardRepresentation(
                goal_dimension=goal_dimension * goal_frames,
                action_dimension=action_length,
                z_dimension=z_dimension,
                backward_hidden_dimension=backward_hidden_dimension,
                backward_hidden_layers=backward_hidden_layers,
                device=device,
                backward_activation=backward_activation,
                layernorms=layernorms,
                include_action=False,
                discrete_actions=False,
            )
            self.backward_representation_target = BackwardRepresentation(
                goal_dimension=goal_dimension * goal_frames,
                action_dimension=action_length,
                z_dimension=z_dimension,
                backward_hidden_dimension=backward_hidden_dimension,
                backward_hidden_layers=backward_hidden_layers,
                device=device,
                backward_activation=backward_activation,
                layernorms=layernorms,
                include_action=False,
                discrete_actions=False,
            )

        self._discount = discount
        self.orthonormalisation_coefficient = orthonormalisation_coefficient
        self._device = device
