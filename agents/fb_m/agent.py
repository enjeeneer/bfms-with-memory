# pylint: disable=[E1120, E1123, W0613, W0632]
"""Module defining the forward-backward agent with memory."""

import math
from pathlib import Path
from typing import Tuple, Dict, Optional, Union

import torch
import numpy as np
from collections import deque

from agents.fb_m.models import RecurrentForwardBackwardRepresentation, ActorModel
from agents.base import (
    AbstractAgent,
    Batch,
    GRUActor,
    LSTMActor,
    MLPActor,
    TransformerActor,
    S4DActor,
    make_aug_encoder,
)
from agents.replay_buffer import MemoryEfficientOfflineReplayBuffer
from agents.utils import schedule, get_grad_norm


class MemoryBasedFB(AbstractAgent):
    """
    Forward-backward agent with memory
    """

    def __init__(
        self,
        observation_dims: Union[int, Tuple[int, int, int]],
        observation_type: str,
        action_length: int,
        goal_dimension: int,
        memory_type: str,
        preprocessed_dimension: int,
        postprocessed_dimension: int,
        history_length: int,
        goal_frames: int,
        backward_history_length: int,
        z_dimension: int,
        forward_hidden_dimension: int,
        forward_hidden_layers: int,
        backward_hidden_dimension: int,
        backward_hidden_layers: int,
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        obs_encoder_hidden_dimension: int,
        forward_activation: str,
        backward_activation: str,
        actor_learning_rate: float,
        critic_learning_rate: float,
        learning_rate_coefficient: float,
        orthonormalisation_coefficient: float,
        discount: float,
        batch_size: int,
        num_encoder_layers: int,
        z_mix_ratio: float,
        std_dev_schedule: str,
        tau: float,
        device: torch.device,
        name: str,
        std_dev_clip: float,
        inference_memory: bool,
        gradient_clipping: bool,
        actor_obs_z_encoder: bool,
        boltzmann_temperature: int,
        recurrent_F: bool,
        recurrent_B: bool,
        z_inference_steps: int,
        layernorms: bool = True,
        pooling: Optional[str] = None,
        gru_dimension: Optional[int] = None,
        transformer_dimension: Optional[int] = None,
        transformer_attention: Optional[str] = None,
        s4_dimension: Optional[int] = None,
        transformer_n_heads: Optional[int] = None,
        discrete_actions: bool = False,
    ):

        if observation_type == "pixels":
            augmentation, image_encoder = make_aug_encoder(
                image_wh=64, obs_shape=observation_dims, device=device
            )
            observation_length = image_encoder.repr_dim
            goal_dimension = image_encoder.repr_dim
        else:
            augmentation = torch.nn.Identity()
            image_encoder = torch.nn.Identity()
            observation_length = observation_dims

        super().__init__(
            observation_length=observation_length,
            action_length=action_length,
            name=name,
        )

        self.augmentation = augmentation
        self.image_encoder = image_encoder
        self.observation_type = observation_type

        self.FB = RecurrentForwardBackwardRepresentation(
            observation_length=observation_length,
            action_length=action_length,
            goal_dimension=goal_dimension,
            z_dimension=z_dimension,
            goal_frames=goal_frames,
            forward_hidden_dimension=forward_hidden_dimension,
            forward_hidden_layers=forward_hidden_layers,
            backward_hidden_dimension=backward_hidden_dimension,
            backward_hidden_layers=backward_hidden_layers,
            forward_activation=forward_activation,
            backward_activation=backward_activation,
            orthonormalisation_coefficient=orthonormalisation_coefficient,
            discount=discount,
            device=device,
            layernorms=layernorms,
            memory_type=memory_type,
            preprocessed_dimension=preprocessed_dimension,
            postprocessed_dimension=postprocessed_dimension,
            history_length=history_length,
            backward_history_length=backward_history_length,
            pooling=pooling,
            num_encoder_layers=num_encoder_layers,
            transformer_dimension=transformer_dimension,
            transformer_attention=transformer_attention,
            s4_dimension=s4_dimension,
            transformer_n_heads=transformer_n_heads,
            discrete_actions=discrete_actions,
            recurrent_F=recurrent_F,
            recurrent_B=recurrent_B,
            gru_dimension=gru_dimension,
        )
        self.FB_optimizer = torch.optim.Adam(
            [
                {"params": self.FB.forward_representation.parameters()},
                {
                    "params": self.FB.backward_representation.parameters(),
                    "lr": critic_learning_rate * learning_rate_coefficient,
                },
            ],
            lr=critic_learning_rate,
        )

        if not discrete_actions:
            if recurrent_F:
                if memory_type == "gru":
                    self.actor = GRUActor(
                        observation_length=observation_length,
                        action_length=action_length,
                        z_length=z_dimension,
                        preprocessed_dimension=preprocessed_dimension,
                        postprocessed_dimension=postprocessed_dimension,
                        obs_encoder_hidden_dimension=obs_encoder_hidden_dimension,
                        actor_hidden_dimension=actor_hidden_dimension,
                        actor_hidden_layers=actor_hidden_layers,
                        std_dev_clip=std_dev_clip,
                        device=device,
                        obs_z_encoder=actor_obs_z_encoder,
                        num_layers=num_encoder_layers,
                        gru_dimension=gru_dimension,
                    )
                elif memory_type == "lstm":
                    self.actor = LSTMActor(
                        observation_length=observation_length,
                        action_length=action_length,
                        z_length=z_dimension,
                        preprocessed_dimension=preprocessed_dimension,
                        postprocessed_dimension=postprocessed_dimension,
                        obs_encoder_hidden_dimension=obs_encoder_hidden_dimension,
                        actor_hidden_dimension=actor_hidden_dimension,
                        actor_hidden_layers=actor_hidden_layers,
                        std_dev_clip=std_dev_clip,
                        device=device,
                        obs_z_encoder=actor_obs_z_encoder,
                    )
                elif memory_type == "transformer":
                    self.actor = TransformerActor(
                        observation_length=observation_length,
                        action_length=action_length,
                        z_length=z_dimension,
                        preprocessed_dimension=preprocessed_dimension,
                        postprocessed_dimension=postprocessed_dimension,
                        transformer_dimension=transformer_dimension,
                        transformer_n_heads=transformer_n_heads,
                        num_layers=num_encoder_layers,
                        transformer_attention=transformer_attention,
                        pooling=pooling,
                        history_length=history_length,
                        obs_encoder_hidden_dimension=obs_encoder_hidden_dimension,
                        actor_hidden_dimension=actor_hidden_dimension,
                        actor_hidden_layers=actor_hidden_layers,
                        std_dev_clip=std_dev_clip,
                        device=device,
                        obs_z_encoder=actor_obs_z_encoder,
                    )
                elif memory_type == "s4d":
                    self.actor = S4DActor(
                        observation_length=observation_length,
                        action_length=action_length,
                        z_length=z_dimension,
                        s4_dimension=s4_dimension,
                        preprocessed_dimension=preprocessed_dimension,
                        postprocessed_dimension=postprocessed_dimension,
                        obs_encoder_hidden_dimension=obs_encoder_hidden_dimension,
                        actor_hidden_dimension=actor_hidden_dimension,
                        actor_hidden_layers=actor_hidden_layers,
                        std_dev_clip=std_dev_clip,
                        device=device,
                        num_layers=num_encoder_layers,
                        obs_z_encoder=actor_obs_z_encoder,
                    )
                elif memory_type == "mlp":
                    self.actor = MLPActor(
                        observation_length=observation_length,
                        action_length=action_length,
                        preprocessed_dimension=preprocessed_dimension,
                        postprocessed_dimension=postprocessed_dimension,
                        actor_hidden_dimension=actor_hidden_dimension,
                        actor_hidden_layers=actor_hidden_layers,
                        std_dev_clip=std_dev_clip,
                        device=device,
                        history_length=history_length,
                        encoder_layers=2,
                        obs_z_encoder=actor_obs_z_encoder,
                        z_length=z_dimension,
                        obs_encoder_hidden_dimension=obs_encoder_hidden_dimension,
                    )
                else:
                    raise ValueError(f"Invalid memory type: {memory_type}")

            # if not using recurrent F then actor is also not recurrent
            else:
                self.actor = ActorModel(
                    observation_length=observation_length,
                    action_length=action_length,
                    preprocessed_dimension=preprocessed_dimension,
                    postprocessed_dimension=postprocessed_dimension,
                    z_dimension=z_dimension,
                    actor_hidden_dimension=actor_hidden_dimension,
                    actor_hidden_layers=actor_hidden_layers,
                    actor_activation="relu",
                    gaussian_actor=False,
                    std_dev_clip=std_dev_clip,
                    device=device,
                )

            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=actor_learning_rate
            )

        else:
            # no actor required for discrete action space
            self.actor = None

        # optimisers
        self.FB_optimizer = torch.optim.Adam(
            [
                {"params": self.FB.forward_representation.parameters()},
                {
                    "params": self.FB.backward_representation.parameters(),
                    "lr": critic_learning_rate * learning_rate_coefficient,
                },
            ],
            lr=critic_learning_rate,
        )

        # load weights into target networks
        self.FB.forward_representation_target.load_state_dict(
            self.FB.forward_representation.state_dict()
        )
        self.FB.backward_representation_target.load_state_dict(
            self.FB.backward_representation.state_dict()
        )

        self._device = device
        self._fb_loss_weight = 10.0
        self.batch_size = batch_size
        self._discrete_actions = discrete_actions
        self._z_mix_ratio = z_mix_ratio
        self._tau = tau
        self._recurrent_B = recurrent_B
        self._recurrent_F = recurrent_F
        self.recurrent_actor = recurrent_F
        self._gradient_clipping = gradient_clipping
        self._z_dimension = z_dimension
        self._history_length = history_length
        self.std_dev_schedule = std_dev_schedule
        self._observation_length = observation_length
        self._action_length = action_length
        self._memory_length = self._history_length if inference_memory else 1
        self.inference_memory = inference_memory
        self.actor_obs_z_encoder = actor_obs_z_encoder
        self.memory_type = memory_type
        self.transformer_attention = transformer_attention
        self.z_inference_steps = z_inference_steps
        self.backward_history_length = backward_history_length
        self._boltzmann_temperature = boltzmann_temperature

        # initialise memory with all zeros
        assert self._history_length > 0, "MemoryBasedFB requires non-zero history length."

    def _reset_memory(self, first_observation: torch.Tensor) -> None:
        """
        Initialises memory with duplicates of the first observation,
        and zeroed actions.
        """

        empty_action = torch.zeros(
            1,
            self._action_length if not self._discrete_actions else 1,
            device=self._device,
        )
        self.observation_memory = deque(
            [first_observation] * (self._memory_length - 1),
            maxlen=(self._memory_length - 1),
        )
        self.action_memory = deque(
            [empty_action] * (self._memory_length),
            maxlen=(self._memory_length),
        )

    @torch.no_grad()
    def act(
        self,
        observation: np.ndarray,
        task: np.array,
        step: int,
        previous_obs_action_internal_state: Optional[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
        previous_obs_z_internal_state: Optional[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
        sample: bool = False,
    ) -> Tuple[np.array, torch.Tensor, torch.Tensor]:
        """
        Used at test time to perform zero-shot rollouts.
        Takes observation array from environment, encodes, and selects
        action from actor.
        Args:
            observation: observation array of shape [observation_length]
            task: task array of shape [z_dimension]
            step: current step in env
            previous_obs_action_internal_state:
                    hidden state of obs_action encoder from previous forward pass
            previous_obs_z_internal_state:
                    hidden state of obs_z encoder from previous forward pass
            sample: whether to sample action from actor distribution
        Returns:
            action: action array of shape [action_length]
        """

        observation = torch.as_tensor(
            observation, dtype=torch.float32, device=self._device
        ).unsqueeze(0)

        # reset memory at start of episode
        if step == 0 and self.recurrent_actor:
            self._reset_memory(first_observation=observation)

        z = torch.as_tensor(task, dtype=torch.float32, device=self._device).unsqueeze(0)

        if self.recurrent_actor:
            action, hidden_obs_action_state, hidden_obs_z_state = self._actor_inference(
                observation=observation,
                z=z,
                step=step,
                previous_obs_action_hidden_state=previous_obs_action_internal_state,
                previous_obs_z_hidden_state=previous_obs_z_internal_state,
                sample=sample,
            )
        else:
            # get action from actor
            std_dev = schedule(self.std_dev_schedule, step)
            h = self.image_encoder(observation)
            action, _ = self.actor(observation=h, z=z, std=std_dev, sample=sample)
            hidden_obs_action_state = None
            hidden_obs_z_state = None

        action = action.detach().cpu().numpy()

        return np.squeeze(action, 0), hidden_obs_action_state, hidden_obs_z_state

    def _actor_inference(
        self,
        observation: torch.Tensor,
        z: torch.Tensor,
        step: int,
        previous_obs_action_hidden_state: Optional[torch.Tensor] = None,
        previous_obs_z_hidden_state: Optional[torch.Tensor] = None,
        sample: bool = False,
    ) -> Tuple[np.array, torch.Tensor, torch.Tensor]:

        # get histories, reshape to [1, history_length, input_dim] and
        # concatenate with current observation
        if self.inference_memory:
            observation_history = torch.stack(
                list(self.observation_memory), dim=1
            ).squeeze(0)
            observation_history = torch.cat(
                [observation_history, observation], dim=0
            ).unsqueeze(
                0
            )  # [1, history_length, obs_dim]
        else:
            observation_history = observation.unsqueeze(1)

        if self.observation_type == "pixels":
            # roll and unroll batch/history dimension for image encoder
            observation_history = self.image_encoder(
                observation_history.view(-1, 3, 64, 64)
            ).view(1, -1, self.observation_length)

        action_history = (
            torch.stack(list(self.action_memory), dim=0).squeeze(1).unsqueeze(0)
        )

        # get z history
        z_history = z.unsqueeze(1).repeat(1, self._memory_length, 1)

        # get first time index for positional encoding (only required for transformer)
        if self.memory_type in ["transformer"]:
            first_time_idx = torch.as_tensor(
                [step], device=self._device, dtype=torch.int
            ).unsqueeze(0)
        else:
            first_time_idx = None

        # inference for discrete actions comes directly from F
        if self._discrete_actions:

            # get Qs for all actions
            action_samples = torch.arange(
                self._action_length, device=self._device
            ).reshape(1, 1, -1, 1)

            F1s, F2s, hiddens = [], [], []
            for i in range(self._action_length):
                (
                    F1,
                    F2,
                    hidden_obs_action_state,
                ) = self.FB.forward_representation.forward(
                    observation_history=observation_history,
                    z_history=z_history,
                    action_history=action_samples[
                        :, :, i, :
                    ],  # batch x history x action
                    first_time_idx=first_time_idx,
                    prev_hidden_state=previous_obs_action_hidden_state,
                )
                F1s.append(F1)
                F2s.append(F2)
                hiddens.append(hidden_obs_action_state)

            F1s = torch.cat(F1s).view(1, self._z_dimension, self._action_length)
            F2s = torch.cat(F2s).view(1, self._z_dimension, self._action_length)
            hiddens = torch.cat(hiddens).view(1, 1, -1, self._action_length)

            Q1, Q2 = [torch.einsum("sda, sd -> sa", Fi, z) for Fi in [F1s, F2s]]
            Q = torch.min(Q1, Q2)

            # get the argmax of the Q values
            idx = torch.argmax(Q, dim=1)

            action = Q[:, idx]
            hidden_obs_action_state = hiddens[:, :, :, idx].squeeze(-1)

            if sample:
                action = (
                    torch.randint_like(action, self.action_length)
                    if np.random.rand() < self._exploration_epsilon
                    else action
                )
            hidden_obs_z_state = None

        # inference for continuous actions comes from actor
        else:
            std_dev = schedule(self.std_dev_schedule, step)
            # get action from actor
            (action, hidden_obs_action_state, hidden_obs_z_state,) = self.actor.forward(
                observation_history=observation_history,
                action_history=action_history,
                z_history=z_history,
                std=std_dev,
                sample=sample,
                prev_hidden_obs_action_state=previous_obs_action_hidden_state,
                prev_hidden_obs_z_state=previous_obs_z_hidden_state,
                first_time_idx=first_time_idx,
            )

        # update memory with current observation and action
        if self.inference_memory:
            self.observation_memory.append(observation)
        self.action_memory.append(action)

        # set hidden_state to None if not carrying hidden states through episode
        if self.inference_memory:
            hidden_obs_action_state = None
            hidden_obs_z_state = None

        return action, hidden_obs_action_state, hidden_obs_z_state

    def _aug_and_encode(self, obs: torch.Tensor) -> torch.Tensor:

        # reshape to [-1, 3, 64, 64]
        if self.observation_type == "pixels":
            obs = obs.view(-1, 3, 64, 64)

        augmented_obs = self.augmentation(obs)
        encoded_obs = self.image_encoder(augmented_obs)

        # back to original shape
        if self.observation_type == "pixels":
            final_obs = encoded_obs.view(obs.shape[0], obs.shape[1], -1)
        else:
            final_obs = encoded_obs

        return final_obs

    def update(self, batch: Batch, step: int) -> Dict[str, float]:
        """
        Updates agent's networks given a batch_size sample from the replay buffer.
        Args:
            batch: memory buffer containing transitions
            step: no. of steps taken in the environment
        Returns:
            metrics: dictionary of metrics for logging
        """

        # augment and encode observations
        observation_histories = self._aug_and_encode(batch.observation_histories)
        next_observation_histories = self._aug_and_encode(
            batch.next_observation_histories
        )
        # extract the correct goals
        if self._recurrent_B:
            goals = batch.goal_histories
            next_goals = batch.next_goal_histories
            goal_positions = batch.goal_positions
            next_goal_positions = batch.next_goal_positions
        else:
            goals = batch.goals
            next_goals = batch.next_goals
            goal_positions = None
            next_goal_positions = None
        backward_goals = self._aug_and_encode(goals.unsqueeze(1)).squeeze(1)
        next_goals = self._aug_and_encode(next_goals.unsqueeze(1)).squeeze(1)
        next_observation_histories = next_observation_histories.detach()
        backward_goals = backward_goals.detach()
        if self._recurrent_B:
            backward_actions = batch.goal_action_histories.detach()

        # sample zs
        zs = self._sample_z(size=self.batch_size)
        perm = torch.randperm(self.batch_size)
        mix_indices = np.where(np.random.rand(self.batch_size) < self._z_mix_ratio)[0]
        backward_goals = backward_goals[perm]
        backward_goals = backward_goals[mix_indices]
        if self._recurrent_B:
            backward_actions = backward_actions[perm]
            backward_actions = backward_actions[mix_indices]
            goal_positions = goal_positions[perm]
            goal_positions = goal_positions[mix_indices]
        else:
            backward_actions = None
            goal_positions = None

        with torch.no_grad():
            if self._recurrent_B:
                mix_zs = self.FB.backward_representation(
                    goal=backward_goals,
                    action=backward_actions,
                    first_time_idx=goal_positions,
                ).detach()
            else:
                mix_zs = self.FB.backward_representation(goal=backward_goals).detach()

        zs[mix_indices] = mix_zs

        # curate z histories
        z_histories = zs.unsqueeze(1).repeat(1, batch.observation_histories.shape[1], 1)

        fb_loss, fb_metrics, _, _, _, _, _, _, _, _ = self._update_fb_inner(
            observation_histories=observation_histories,
            next_observation_histories=next_observation_histories,
            action_histories=batch.action_histories.to(torch.int64)
            if self._discrete_actions
            else batch.action_histories,
            next_action_histories=batch.next_action_histories.to(torch.int64)
            if self._discrete_actions
            else batch.next_action_histories,
            next_goals=next_goals,
            goal_first_time_idxs=next_goal_positions,
            discounts=batch.discounts,
            z_histories=z_histories,
            step=step,
            first_time_idxs=batch.positions,
            next_first_time_idxs=batch.next_positions
            if self.memory_type in ["transformer"]
            else None,
            next_goal_actions=batch.next_goal_action_histories,
        )

        self.FB_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()
        for param in self.FB.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.FB_optimizer.step()

        # update actor
        if not self._discrete_actions:
            actor_loss, actor_metrics = self._update_actor(
                observation_histories=next_observation_histories.detach(),
                z_histories=z_histories,
                action_histories=batch.action_histories,
                step=step,
                first_time_idxs=batch.positions,
            )
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            for param in self.actor.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            self.actor_optimizer.step()
            grad_metrics = {}

        else:
            actor_metrics = {}
            grad_metrics = {}

        # update target networks for forwards and backwards models
        self.soft_update_params(
            network=self.FB.forward_representation,
            target_network=self.FB.forward_representation_target,
            tau=self._tau,
        )
        self.soft_update_params(
            network=self.FB.backward_representation,
            target_network=self.FB.backward_representation_target,
            tau=self._tau,
        )

        metrics = {
            **fb_metrics,
            **actor_metrics,
            **grad_metrics,
        }

        return metrics

    def _update_fb_inner(
        self,
        observation_histories: torch.Tensor,
        next_observation_histories: torch.Tensor,
        action_histories: torch.Tensor,
        next_action_histories: torch.Tensor,
        next_goals: torch.Tensor,
        next_goal_actions: torch.Tensor,
        goal_first_time_idxs: torch.Tensor,
        discounts: torch.Tensor,
        z_histories: torch.Tensor,
        first_time_idxs: torch.Tensor,
        next_first_time_idxs: torch.Tensor,
        step: int,
    ):
        """
        Loss computation common to FB and all child classes. All equation references
        are to the appendix of the FB paper (Touati et. al (2022)).
        The loss contains two components:
            1. Forward-backward representation loss: a Bellman update on the successor
                measure (equation 24, Appendix B)
            2. Orthonormalisation loss: constrains backward function such that the
                measure of state s from state s = 1 (equation 26, Appendix B)
            Note: Q loss (Equation 9) is not implemented.
        Args:
            observation_histories: observation tensor of shape [batch_size,
                                            history_length, observation_length]
            next_observation_histories: next observation tensor of shape [batch_size,
                                            history_length, observation_length]
            action_histories: action tensor of shape [batch_size,
                                                history_length, action_length]
            next_goals: next goal tensor of shape [batch_size, goal_length]
            next_goal_actions: action sequence for Backward model (only used if B(s,a))
            goal_first_time_idxs: tensor of shape [batch_size, 1]
            discounts: discount tensor of shape [batch_size, 1]
            z_histories: task tensor of shape [batch_size, history_length, z_dimension]
            first_time_idxs: tensor of shape [batch_size, 1] containing the index of the
                            first observation in the histories
            step: current training step
        Returns:
            total_loss: total loss for FB
            metrics: dictionary of metrics for logging
            F1: forward embedding no. 1
            F2: forward embedding no. 2
            B_next: backward embedding
            M1_next: successor measure no. 1
            M2_next: successor measure no. 2
            target_B: backward embedding from target network
            off_diagonal: off-diagonal masking array
            actor_std_dev: current actor standard deviation
        """

        if self._discrete_actions:
            assert action_histories.shape == (
                self.batch_size,
                self._history_length,
                1,
            ), (
                f"Batch of discrete actions in dataset have shape "
                f"{action_histories.shape}, but"
                f"this implementation expects shape (batch_size, history_length, 1)."
            )

        with torch.no_grad():

            if self._discrete_actions:

                # we initially pass all possible actions through the
                # forward model to get the Q values for each action

                # tile actions discrete actions
                next_action_histories_tile = (
                    next_action_histories.unsqueeze(1)
                    .repeat(1, self._action_length, 1, 1)
                    .to(self._device)
                )
                next_action_samples = (
                    torch.arange(self._action_length)
                    .repeat(self.batch_size, 1, 1)
                    .to(self._device)
                    .view(self.batch_size, self._action_length, 1)
                )

                # add samples to action history
                next_action_histories_tile[:, :, -1, :] = next_action_samples

                targets_1, targets_2 = [], []
                for i in range(self._action_length):
                    (
                        target_F1,
                        target_F2,
                        _,
                    ) = self.FB.forward_representation_target.forward(
                        observation_history=next_observation_histories,
                        z_history=z_histories,
                        action_history=next_action_histories_tile[
                            :, i, :, :
                        ],  # batch x history x action
                        first_time_idx=next_first_time_idxs,
                    )
                    targets_1.append(target_F1)
                    targets_2.append(target_F2)

                target_F1 = torch.cat(targets_1).view(
                    self.batch_size, self._z_dimension, self._action_length
                )
                target_F2 = torch.cat(targets_2).view(
                    self.batch_size, self._z_dimension, self._action_length
                )

                next_Q1, next_Q2 = [
                    torch.einsum("sda, sd -> sa", Fi, z_histories[:, -1, :])
                    for Fi in [target_F1, target_F2]
                ]

                next_Q = torch.min(next_Q1, next_Q2)

                # take the softmax of the Q values
                if self._boltzmann_temperature > 0:
                    pi = torch.nn.functional.softmax(
                        next_Q / self._boltzmann_temperature, dim=-1
                    )

                    target_F1, target_F2 = [
                        torch.einsum("sa, sda -> sd", pi, Fi)
                        for Fi in [target_F1, target_F2]
                    ]  # batch x z_dim

                # take the argmax of the Q values
                else:
                    pi = next_Q.max(1)[1].unsqueeze(1)
                    target_F1, target_F2 = [
                        Fi.gather(
                            -1, pi.repeat(1, self._z_dimension)[:, :, None]
                        ).squeeze()
                        for Fi in [target_F1, target_F2]
                    ]

                actor_std_dev = 0.0

            else:
                # sample actions from actor
                actor_std_dev = schedule(self.std_dev_schedule, step)
                next_encoded_history = None

                if self.recurrent_actor:
                    next_actions, _, _ = self.actor.forward(
                        observation_history=next_observation_histories,
                        action_history=action_histories,
                        z_history=z_histories,
                        std=actor_std_dev,
                        sample=True,
                        first_time_idx=next_first_time_idxs,
                        encoded_history=next_encoded_history,
                    )

                    # update action_history by concatenating next action
                    # and removing first action
                    # action histories shape:
                    # [batch_size, history_length, action_length]
                    # next actions shape: [batch_size, action_length]
                    next_action_histories = torch.cat(
                        [action_histories[:, 1:, :], next_actions.unsqueeze(1)], dim=1
                    )

                    (
                        target_F1,
                        target_F2,
                        _,
                    ) = self.FB.forward_representation_target.forward(
                        observation_history=next_observation_histories,
                        z_history=z_histories,
                        action_history=next_action_histories,
                        first_time_idx=next_first_time_idxs,
                    )

                # normal actor
                else:
                    next_actions, _ = self.actor(
                        next_observation_histories[:, -1, :],
                        z_histories[:, -1, :],
                        actor_std_dev,
                        sample=True,
                    )
                    target_F1, target_F2 = self.FB.forward_representation_target(
                        observation=next_observation_histories[:, -1, :],
                        z=z_histories[:, -1, :],
                        action=next_actions,
                    )

            target_B = self.FB.backward_representation_target.forward(
                goal=next_goals,
                action=next_goal_actions,
                first_time_idx=goal_first_time_idxs,
            )
            target_M1 = torch.einsum(
                "sd, td -> st", target_F1, target_B
            )  # [batch_size, batch_size]
            target_M2 = torch.einsum(
                "sd, td -> st", target_F2, target_B
            )  # [batch_size, batch_size]
            target_M = torch.min(target_M1, target_M2)

        # --- Forward-backward representation loss ---
        if self._recurrent_F:
            F1, F2, _ = self.FB.forward_representation.forward(
                observation_history=observation_histories,
                action_history=action_histories,
                z_history=z_histories,
                first_time_idx=first_time_idxs,
            )
        else:
            F1, F2 = self.FB.forward_representation.forward(
                observation=observation_histories[:, -1, :],
                action=action_histories[:, -1, :],
                z=z_histories[:, -1, :],
            )

        B_next = self.FB.backward_representation.forward(
            goal=next_goals,
            action=next_goal_actions,
            first_time_idx=goal_first_time_idxs,
        )

        M1_next = torch.einsum("sd, td -> st", F1, B_next)
        M2_next = torch.einsum("sd, td -> st", F2, B_next)

        # diagonal/off diagonal method mentioned here:
        # https://github.com/facebookresearch/controllable_agent/issues/4
        I = torch.eye(*M1_next.size(), device=self._device)  # next state = s_{t+1}
        off_diagonal = ~I.bool()  # future states =/= s_{t+1}

        fb_off_diag_loss = 0.5 * sum(
            (M - discounts * target_M)[off_diagonal].pow(2).mean()
            for M in [M1_next, M2_next]
        )

        fb_diag_loss = -sum(M.diag().mean() for M in [M1_next, M2_next])

        fb_loss = fb_diag_loss + fb_off_diag_loss

        # --- orthonormalisation loss ---
        covariance = torch.matmul(B_next, B_next.T)
        ortho_loss_diag = -2 * covariance.diag().mean()
        ortho_loss_off_diag = covariance[off_diagonal].pow(2).mean()
        ortho_loss = self.FB.orthonormalisation_coefficient * (
            ortho_loss_diag + ortho_loss_off_diag
        )

        total_loss = fb_loss + ortho_loss

        metrics = {
            "train/forward_backward_total_loss": total_loss,
            "train/forward_backward_fb_loss": fb_loss,
            "train/forward_backward_fb_diag_loss": fb_diag_loss,
            "train/forward_backward_fb_off_diag_loss": fb_off_diag_loss,
            "train/ortho_diag_loss": ortho_loss_diag,
            "train/ortho_off_diag_loss": ortho_loss_off_diag,
            "train/target_M": target_M.mean().item(),
            "train/M": M1_next.mean().item(),
            "train/F": F1.mean().item(),
            "train/F_max": F1.max().item(),
            "train/F_min": F1.min().item(),
            "train/B": B_next.mean().item(),
        }

        return (
            total_loss,
            metrics,
            F1,
            F2,
            B_next,
            M1_next,
            M2_next,
            target_B,
            off_diagonal,
            actor_std_dev,
        )

    def _update_actor(
        self,
        observation_histories: torch.Tensor,
        action_histories: torch.Tensor,
        z_histories: torch.Tensor,
        first_time_idxs: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the actor loss.
        Args:
            observation_histories: tensor of shape [batch_size,
                                        history_length, observation_length]
            z_histories: tensor of shape [batch_size,
                                            history_length, z_dimension]
            action_histories: tensor of shape [batch_size,
                                            history_length, action_length]
            step: current training step
        Returns:
            metrics: dictionary of metrics for logging
        """
        std = schedule(self.std_dev_schedule, step)

        encoded_history = None

        if self.recurrent_actor:
            # stagger observations and actions
            actions, _, _ = self.actor.forward(
                observation_history=observation_histories,
                action_history=action_histories,
                z_history=z_histories,
                encoded_history=encoded_history,
                std=std,
                sample=True,
                first_time_idx=first_time_idxs,
            )
        else:
            actions, _ = self.actor(
                observation=observation_histories[:, -1, :],
                z=z_histories[:, -1, :],
                std=std,
                sample=True,
            )

        # update action history by replacing final action with action from actor
        # add new action to action history and remove first action
        actions = actions.unsqueeze(1)
        action_histories = torch.cat([action_histories[:, 1:, :], actions], dim=1)

        if self._recurrent_F:
            F1, F2, _ = self.FB.forward_representation.forward(
                observation_history=observation_histories,
                action_history=action_histories,
                z_history=z_histories,
                first_time_idx=first_time_idxs,
            )
        else:
            F1, F2 = self.FB.forward_representation(
                observation=observation_histories[:, -1, :],
                z=z_histories[:, -1, :],
                action=action_histories[:, -1, :],
            )

        # get final z from history of repeated zs for calculating Q
        z = z_histories[:, -1, :]

        # get Qs from F and z
        Q1 = torch.einsum("sd, sd -> s", F1, z)
        Q2 = torch.einsum("sd, sd -> s", F2, z)
        Q = torch.min(Q1, Q2)

        # update actor towards action that maximise Q (minimise -Q)
        actor_loss = -Q

        actor_loss = actor_loss.mean()

        metrics = {
            "train/actor_loss": actor_loss.item(),
            "train/actor_Q": Q.mean().item(),
        }

        return actor_loss, metrics

    def load(self, filepath: Path):
        """Loads model."""
        pass

    def _sample_z(self, size: int) -> torch.Tensor:
        """Samples z in the sphere of radius sqrt(D)."""
        gaussian_random_variable = torch.randn(
            size, self._z_dimension, dtype=torch.float32, device=self._device
        )
        gaussian_random_variable = torch.nn.functional.normalize(
            gaussian_random_variable, dim=1
        )
        z = math.sqrt(self._z_dimension) * gaussian_random_variable

        return z

    def infer_z(
        self,
        replay_buffer: MemoryEfficientOfflineReplayBuffer,
        multiplier: int,
        goal_state_dict: Optional[dict] = None,
        popgym: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Infers z from observations and rewards. Follows page 17 from:
        https://arxiv.org/pdf/2209.14935.pdf. If no rewards are passed, we
        assume we're inferring z from a goal-reaching task, and thus set z
        directly from the goal state.
        Args:
            replay_buffer: buffer for goal sampling
            goal_state_dict: goal state for z inference
        Returns:
            z: tensor of shape [z_dimension,]
        """
        multipliers = (multiplier, multiplier)
        zs = {}
        if goal_state_dict is not None:
            for task, goal_state in goal_state_dict.items():
                with torch.no_grad():
                    goals = self.image_encoder(goal_state)

                # hard code the actions as zeros (i.e. no action required
                # once at goal state)
                actions = torch.zeros(
                    1,
                    self.backward_history_length,
                    self._action_length,
                    device=self._device,
                )

                # hard code first time idx as terminal step - history_length
                # terminal step is always 1000
                first_time_idx = torch.tile(
                    torch.tensor(
                        1000 - self.backward_history_length,
                        device=self._device,
                        dtype=torch.int,
                    ),
                    (1, self._history_length, 1),
                )

                with torch.no_grad():
                    z = self.FB.backward_representation.forward(
                        goals,
                        actions,
                        first_time_idx,
                    )
                z = math.sqrt(self._z_dimension) * torch.nn.functional.normalize(
                    z, dim=1
                )
                z = z.squeeze().cpu().numpy()
                zs[task] = z
        else:
            z_inference_chunks = int(self.z_inference_steps / 10000)
            # sample goals/rewards/positions
            for _ in range(z_inference_chunks):
                (
                    goals,
                    rewards,
                    positions,
                    actions,
                ) = replay_buffer.sample_task_inference_transitions(
                    inference_steps=10000,
                    popgym=popgym,
                )
                goals = goals[multipliers]
                rewards = rewards[multipliers]
                positions = positions[multipliers]
                actions = actions[multipliers]

                with torch.no_grad():
                    goals = self.image_encoder(goals)

                with torch.no_grad():
                    for task, reward in rewards.items():
                        if task not in zs:
                            zs[task] = []
                        z = self.FB.backward_representation.forward(
                            goal=goals, action=actions, first_time_idx=positions
                        )
                        z = (
                            torch.matmul(reward.T, z) / reward.shape[0]
                        )  # reward-weighted average

                        z = math.sqrt(
                            self._z_dimension
                        ) * torch.nn.functional.normalize(z, dim=1)

                        z = z.squeeze().cpu().numpy()

                        zs[task].append(z)

            for task in zs:
                zs[task] = np.mean(zs[task], axis=0)

        return zs

    @staticmethod
    def soft_update_params(
        network: torch.nn.Sequential, target_network: torch.nn.Sequential, tau: float
    ) -> None:
        """
        Soft updates the target network parameters via Polyak averaging.
        Args:
            network: Online network.
            target_network: Target network.
            tau: Interpolation parameter.
        """

        for param, target_param in zip(
            network.parameters(), target_network.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def _get_grad_norms(self):
        """
        Returns the gradient norms of the forward and backward representations.
        """
        grads = {
            "train/actor grad norm": get_grad_norm(self.actor),
            "train/FB grad norm": get_grad_norm(self.FB),
        }

        return grads
