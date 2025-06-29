# pylint: disable-all
"""Module defining the Recurrent Successor Feature Agent."""

import math
from pathlib import Path
from typing import Tuple, Dict, Union, Optional

import torch
import numpy as np
from collections import deque

from agents.fb_m.models import ActorModel, ForwardRepresentation
from agents.fb_m.models import RecurrentForwardRepresentation
from agents.hilp_m.models import RecurrentHILPFeatures
from agents.hilp_m.models import HILPFeatures
from agents.base import (
    AbstractAgent,
    Batch,
    make_aug_encoder,
    MemoryEfficientOfflineReplayBuffer,
    MLPActor,
)
from agents.utils import schedule


class MemoryBasedHILP(AbstractAgent):
    """
    HILP agent with memory.
    """

    def __init__(
        self,
        observation_dims: int,
        goal_dimension: int,
        action_length: int,
        observation_type: str,
        preprocessed_dimension: int,
        postprocessed_dimension: int,
        z_dimension: int,
        forward_hidden_dimension: int,
        forward_hidden_layers: int,
        features_hidden_dimension: int,
        features_hidden_layers: int,
        features_activation: str,
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        forward_activation: str,
        actor_activation: str,
        actor_learning_rate: float,
        sf_learning_rate: float,
        feature_learning_rate: float,
        batch_size: int,
        gaussian_actor: bool,
        std_dev_clip: float,
        obs_encoder_hidden_dimension: int,
        std_dev_schedule: str,
        tau: float,
        device: torch.device,
        features: str,
        frames: int,
        z_inference_steps: int,
        name: str,
        hilp_discount: float,
        hilp_iql_expectile: float,
        z_mix_ratio: float,
        q_loss: bool,
        history_length: int,
        memory_type: str,
        pooling: str,
        transformer_dimension: int,
        transformer_attention: str,
        transformer_n_heads: int,
        gru_dimension: int,
        s4_dimension: int,
        num_encoder_layers: int,
        phi_history_length: int,
        recurrent_F: bool,
        inference_memory: bool,
        recurrent_phi: bool,
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
            name=name + f"-{features}",
        )

        self.augmentation = augmentation
        self.image_encoder = image_encoder
        self.observation_type = observation_type

        self._recurrent_F = recurrent_F
        if self._recurrent_F:
            self.successor_net = RecurrentForwardRepresentation(
                observation_length=observation_length,
                action_length=action_length,
                preprocessed_dimension=preprocessed_dimension,
                postprocessed_dimension=postprocessed_dimension,
                memory_type=memory_type,
                history_length=history_length,
                z_dimension=z_dimension,
                forward_hidden_dimension=forward_hidden_dimension,
                forward_hidden_layers=forward_hidden_layers,
                num_encoder_layers=num_encoder_layers,
                obs_z_encoder=False,
                shared_encoder=False,
                pooling=pooling,
                transformer_dimension=transformer_dimension,
                gru_dimension=gru_dimension,
                transformer_attention=transformer_attention,
                s4_dimension=s4_dimension,
                transformer_n_heads=transformer_n_heads,
                forward_activation=forward_activation,
                device=device,
                discrete_actions=False,  # discrete SF not implemented
            )
            self.successor_net_target = RecurrentForwardRepresentation(
                observation_length=observation_length,
                action_length=action_length,
                preprocessed_dimension=preprocessed_dimension,
                postprocessed_dimension=postprocessed_dimension,
                memory_type=memory_type,
                history_length=history_length,
                z_dimension=z_dimension,
                forward_hidden_dimension=forward_hidden_dimension,
                forward_hidden_layers=forward_hidden_layers,
                num_encoder_layers=num_encoder_layers,
                obs_z_encoder=False,
                shared_encoder=False,
                pooling=pooling,
                transformer_dimension=transformer_dimension,
                gru_dimension=gru_dimension,
                transformer_attention=transformer_attention,
                s4_dimension=s4_dimension,
                transformer_n_heads=transformer_n_heads,
                forward_activation=forward_activation,
                device=device,
                discrete_actions=False,  # discrete SF not implemented
            )
            self.actor = MLPActor(
                observation_length=observation_length,
                action_length=action_length,
                preprocessed_dimension=preprocessed_dimension,
                postprocessed_dimension=postprocessed_dimension,
                z_length=z_dimension,
                actor_hidden_dimension=actor_hidden_dimension,
                actor_hidden_layers=actor_hidden_layers,
                history_length=history_length,
                encoder_layers=2,
                obs_z_encoder=False,
                obs_encoder_hidden_dimension=obs_encoder_hidden_dimension,
                std_dev_clip=std_dev_clip,
                device=device,
            )
        else:
            self.successor_net = ForwardRepresentation(
                observation_length=observation_length,
                action_length=action_length,
                preprocessed_dimension=preprocessed_dimension,
                postprocessed_dimension=postprocessed_dimension,
                z_dimension=z_dimension,
                forward_hidden_dimension=forward_hidden_dimension,
                forward_hidden_layers=forward_hidden_layers,
                forward_activation=forward_activation,
                device=device,
                discrete_actions=False,  # discrete SF not implemented
                layernorms=True,
            )
            self.successor_net_target = ForwardRepresentation(
                observation_length=observation_length,
                action_length=action_length,
                preprocessed_dimension=preprocessed_dimension,
                postprocessed_dimension=postprocessed_dimension,
                z_dimension=z_dimension,
                forward_hidden_dimension=forward_hidden_dimension,
                forward_hidden_layers=forward_hidden_layers,
                forward_activation=forward_activation,
                device=device,
                discrete_actions=False,  # discrete SF not implemented
                layernorms=True,
            )
            self.actor = ActorModel(
                observation_length=observation_length,
                action_length=action_length,
                preprocessed_dimension=preprocessed_dimension,
                postprocessed_dimension=postprocessed_dimension,
                z_dimension=z_dimension,
                actor_hidden_dimension=actor_hidden_dimension,
                actor_hidden_layers=actor_hidden_layers,
                actor_activation=actor_activation,
                gaussian_actor=gaussian_actor,
                std_dev_clip=std_dev_clip,
                device=device,
            )

        self.successor_net_target.load_state_dict(self.successor_net.state_dict())
        self.features = features
        self._recurrent_phi = recurrent_phi

        if features == "hilp":
            if self._recurrent_phi:
                self.feature_net = RecurrentHILPFeatures(
                    goal_dimension=goal_dimension,
                    action_length=action_length,
                    z_dimension=z_dimension,
                    hidden_dimension=features_hidden_dimension,
                    num_encoder_layers=num_encoder_layers,
                    device=device,
                    memory_type=memory_type,
                    history_length=phi_history_length,
                    discount=hilp_discount,
                    iql_expectile=hilp_iql_expectile,
                )

            else:
                self.feature_net = HILPFeatures(
                    observation_dim=goal_dimension,
                    z_dimension=z_dimension,
                    hidden_dimension=features_hidden_dimension,
                    hidden_layers=features_hidden_layers,
                    device=device,
                    layernorm=True,
                    activation=features_activation,
                    discount=hilp_discount,
                    iql_expectile=hilp_iql_expectile,
                )
        else:
            raise ValueError(f"Unknown feature type: {features}")

        self.encoder = torch.nn.Identity()
        self.augmentation = torch.nn.Identity()

        # optimisers
        self.sf_optimizer = torch.optim.Adam(
            self.successor_net.parameters(),
            lr=sf_learning_rate,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate
        )
        self.feature_optimizer = torch.optim.Adam(
            self.feature_net.parameters(), lr=feature_learning_rate
        )

        self._device = device
        self._q_loss = q_loss
        self.batch_size = batch_size
        self._tau = tau
        self._z_dimension = z_dimension
        self._z_mix_ratio = z_mix_ratio
        self._history_length = history_length
        self._memory_length = self._history_length if inference_memory else 1
        self.inference_memory = inference_memory
        self.std_dev_schedule = std_dev_schedule
        self._shared_encoder = False
        self._discrete_actions = False
        self._action_length = action_length
        self.recurrent_actor = recurrent_F
        self.z_inference_steps = z_inference_steps
        self.frames = frames
        self.inv_cov = torch.eye(z_dimension, dtype=torch.float32, device=device)
        self.observation_type = observation_type
        self.memory_type = memory_type
        self.backward_history_length = phi_history_length

    def _reset_frames(self, first_observation: torch.Tensor) -> None:
        """
        Initialises memory with the first observation repeated.
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
    ) -> Tuple[np.array, torch.Tensor]:
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
            std_dev: current actor standard deviation
        """

        observation = torch.as_tensor(
            observation, dtype=torch.float32, device=self._device
        ).unsqueeze(0)

        # reset memory at start of episode
        if step == 0 and self.recurrent_actor:
            self._reset_frames(first_observation=observation)

        z = torch.as_tensor(task, dtype=torch.float32, device=self._device).unsqueeze(0)

        if self.recurrent_actor:
            action, hidden_obs_action_state = self._actor_inference(
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

        action = action.detach().cpu().numpy()

        return np.squeeze(action, 0), hidden_obs_action_state

    def _actor_inference(
        self,
        observation: torch.Tensor,
        z: torch.Tensor,
        step: int,
        previous_obs_action_hidden_state: Optional[torch.Tensor] = None,
        previous_obs_z_hidden_state: Optional[torch.Tensor] = None,
        sample: bool = False,
    ) -> Tuple[np.array, torch.Tensor]:

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

        std_dev = schedule(self.std_dev_schedule, step)
        # get action from actor
        (action, hidden_obs_action_state, _,) = self.actor.forward(
            observation_history=observation_history,
            action_history=action_history,
            z_history=z_history,
            std=std_dev,
            sample=sample,
            prev_hidden_obs_action_state=previous_obs_action_hidden_state,
            prev_hidden_obs_z_state=previous_obs_z_hidden_state,
            first_time_idx=None,
        )

        # update memory with current observation and action
        if self.inference_memory:
            self.observation_memory.append(observation)
        self.action_memory.append(action)

        # set hidden_state to None if not carrying hidden states through episode
        if self.inference_memory:
            hidden_obs_action_state = None

        return action, hidden_obs_action_state

    def update(self, batch: Batch, step: int) -> Dict[str, float]:
        """
        Updates agent's networks given a batch_size sample from the replay buffer.
        Args:
            batch: memory buffer containing transitions
            step: no. of steps taken in the environment
        Returns:
            metrics: dictionary of metrics for logging
        """

        # sample zs and betas if appropriate
        zs = self._sample_z(size=self.batch_size)

        if self._recurrent_phi:
            goals = batch.goal_histories
            next_goals = batch.next_goal_histories
            future_goals = batch.future_goal_histories
        else:
            goals = batch.goals
            next_goals = batch.next_goals
            future_goals = batch.future_goals

        if self._z_mix_ratio > 0:
            perm = torch.randperm(self.batch_size)

            if self._recurrent_phi:
                next_goal_histories = next_goals[perm]
                next_goal_action_histories = batch.next_goal_action_histories[perm]
                with torch.no_grad():
                    phi = self.feature_net.forward(
                        goal=next_goal_histories, action=next_goal_action_histories
                    )
            else:
                phi_input = next_goals[perm]
                with torch.no_grad():
                    phi = self.feature_net.forward(phi_input)

            # compute inverse of cov of phi
            cov = torch.matmul(phi.T, phi) / phi.shape[0]
            inv_cov = torch.linalg.pinv(cov)

            mix_idxs = np.where(
                np.random.uniform(size=self.batch_size) < self._z_mix_ratio
            )[0]

            with torch.no_grad():
                new_z = phi[mix_idxs]

            new_z = torch.matmul(new_z, inv_cov)  # batch_size x z_dim
            new_z = math.sqrt(self._z_dimension) * torch.nn.functional.normalize(
                new_z, dim=1
            )
            zs[mix_idxs] = new_z

        # curate z histories
        z_histories = zs.unsqueeze(1).repeat(1, batch.observation_histories.shape[1], 1)

        # update successor nets
        sf_metrics = self.update_successor_features(
            observation_histories=batch.observation_histories,
            next_observation_histories=batch.next_observation_histories,
            goal_histories=goals,
            next_goal_histories=next_goals,
            future_goal_histories=future_goals,
            action_histories=batch.action_histories,
            goal_action_histories=batch.goal_action_histories,
            next_goal_action_histories=batch.next_goal_action_histories,
            future_goal_action_histories=batch.future_goal_action_histories,
            z_histories=z_histories,
            discounts=batch.discounts,
            step=step,
        )

        # update actor (next observations because actions/observation staggered)
        actor_metrics = self.update_actor(
            observation_histories=batch.next_observation_histories.detach(),
            action_histories=batch.action_histories,
            z_histories=z_histories,
            step=step,
        )

        # update target networks for successor features
        self.soft_update_params(
            network=self.successor_net,
            target_network=self.successor_net_target,
            tau=self._tau,
        )

        metrics = {
            **sf_metrics,
            **actor_metrics,
        }

        return metrics

    def update_successor_features(
        self,
        observation_histories: torch.Tensor,
        next_observation_histories: torch.Tensor,
        goal_histories: torch.Tensor,
        next_goal_histories: torch.Tensor,
        future_goal_histories: torch.Tensor,
        action_histories: torch.Tensor,
        goal_action_histories: torch.Tensor,
        next_goal_action_histories: torch.Tensor,
        future_goal_action_histories: torch.Tensor,
        z_histories: torch.Tensor,
        discounts: torch.Tensor,
        step: int,
    ) -> Dict[str, float]:
        """
        Updates the successor features.
        Args:
            observations: observation tensor of shape [batch_size, observation_length]
            actions: action tensor of shape [batch_size, action_length]
            next_observations: next observation tensor of
                                shape [batch_size, observation_length]
            future_observations: future observation tensor of
                                shape [batch_size, observation_length]
            discounts: discount tensor of shape [batch_size, 1]
            zs: policy tensor of shape [batch_size, z_dimension]
            step: current training step
        Returns:
            metrics: dictionary of metrics for logging
        """

        sf_loss, sf_metrics = self._get_successor_loss(
            observation_histories=observation_histories,
            action_histories=action_histories,
            next_observation_histories=next_observation_histories,
            next_goal_histories=next_goal_histories,
            next_goal_action_histories=next_goal_action_histories,
            discounts=discounts,
            z_histories=z_histories,
            step=step,
        )
        if self._recurrent_phi:
            phi_loss, phi_metrics = self.feature_net.get_loss(  # pylint: disable=E1123
                goal_histories=goal_histories,
                action_histories=goal_action_histories,
                next_goal_histories=next_goal_histories,
                next_action_histories=next_goal_action_histories,
                future_goal_histories=future_goal_histories,
                future_action_histories=future_goal_action_histories,
            )
        else:
            phi_loss, phi_metrics = self.feature_net.get_loss(  # pylint: disable=E1123
                observations=goal_histories,
                next_observations=next_goal_histories,
                future_observations=future_goal_histories,
            )

        # step optimisers
        self.sf_optimizer.zero_grad(set_to_none=True)
        sf_loss.backward()
        for param in self.successor_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.sf_optimizer.step()

        self.feature_optimizer.zero_grad(set_to_none=True)
        phi_loss.backward(retain_graph=True)
        for param in self.feature_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.feature_optimizer.step()

        metrics = {
            **sf_metrics,
            **phi_metrics,
        }

        return metrics

    def _get_successor_loss(
        self,
        observation_histories: torch.Tensor,
        action_histories: torch.Tensor,
        next_observation_histories: torch.Tensor,
        next_goal_histories: torch.Tensor,
        next_goal_action_histories: torch.Tensor,
        discounts: torch.Tensor,
        z_histories: torch.Tensor,
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
            discounts: discount tensor of shape [batch_size, 1]
            zs: policy tensor of shape [batch_size, z_dimension]
            step: current training step
        Returns:
            total_loss: total loss for FB
            metrics: dictionary of metrics for logging
        """

        with torch.no_grad():
            actor_std_dev = schedule(self.std_dev_schedule, step)
            if self.recurrent_actor:
                next_actions, _, _ = self.actor.forward(
                    observation_history=next_observation_histories,
                    action_history=action_histories,
                    z_history=z_histories,
                    std=actor_std_dev,
                    sample=True,
                    first_time_idx=None,
                    encoded_history=None,
                )
                next_action_histories = torch.cat(
                    [action_histories[:, 1:, :], next_actions.unsqueeze(1)], dim=1
                )

                next_F1, next_F2, _ = self.successor_net_target.forward(
                    observation_history=next_observation_histories,
                    z_history=z_histories,
                    action_history=next_action_histories,
                    first_time_idx=None,
                    obs_action_embedding=None,
                )
            else:
                next_actions, _ = self.actor(
                    next_observation_histories[:, -1, :],
                    z_histories[:, -1, :],
                    actor_std_dev,
                    sample=True,
                )
                next_F1, next_F2 = self.successor_net_target.forward(
                    observation=next_observation_histories[:, -1, :],
                    z=z_histories[:, -1, :],
                    action=next_actions,
                )

            if self._recurrent_phi:
                target_phi = self.feature_net.forward(
                    goal=next_goal_histories, action=next_goal_action_histories
                ).detach()
            else:
                target_phi = self.feature_net.forward(goal=next_goal_histories).detach()
            next_Q1, next_Q2 = [
                torch.einsum("sd, sd -> s", next_Fi, z_histories[:, -1, :])
                for next_Fi in [next_F1, next_F2]
            ]
            next_F = torch.where((next_Q1 < next_Q2).reshape(-1, 1), next_F1, next_F2)
            target_F = target_phi + discounts * next_F

        # --- successor net loss ---
        if self._recurrent_F:
            F1, F2, _ = self.successor_net.forward(
                observation_history=observation_histories,
                action_history=action_histories,
                z_history=z_histories,
                first_time_idx=None,
                obs_action_embedding=None,
            )
        else:
            F1, F2 = self.successor_net.forward(
                observation=observation_histories[:, -1, :],
                action=action_histories[:, -1, :],
                z=z_histories[:, -1, :],
            )
        if self._q_loss:
            Q1, Q2 = [
                torch.einsum("sd, sd -> s", Fi, z_histories[:, -1, :])
                for Fi in [F1, F2]
            ]
            target_Q = torch.einsum("sd, sd -> s", target_F, z_histories[:, -1, :])
            sf_loss = torch.nn.functional.mse_loss(
                Q1, target_Q
            ) + torch.nn.functional.mse_loss(Q2, target_Q)
        else:
            sf_loss = torch.nn.functional.mse_loss(
                F1, target_F
            ) + torch.nn.functional.mse_loss(F2, target_F)

        metrics = {
            "train/sf_loss": sf_loss.item(),
            "train/target_F": target_F.mean().item(),
            "train/F": F1.mean().item(),
            "train/F_max": F1.max().item(),
            "train/F_min": F1.min().item(),
        }

        return sf_loss, metrics

    def update_actor(
        self,
        observation_histories: torch.Tensor,
        action_histories: torch.Tensor,
        z_histories: torch.Tensor,
        step: int,
    ) -> Dict[str, float]:
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
        if self.recurrent_actor:
            # stagger observations and actions
            actions, _, _ = self.actor.forward(
                observation_history=observation_histories,
                action_history=action_histories,
                z_history=z_histories,
                encoded_history=None,
                std=std,
                sample=True,
                first_time_idx=None,
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
            F1, F2, _ = self.successor_net.forward(
                observation_history=observation_histories,
                action_history=action_histories,
                obs_action_embedding=None,
                z_history=z_histories,
                first_time_idx=None,
            )
        else:
            F1, F2 = self.successor_net.forward(
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

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()

        metrics = {
            "train/actor_loss": actor_loss.item(),
            "train/actor_Q": Q.mean().item(),
        }

        return metrics

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

    def _sample_betas(self, size: int) -> torch.Tensor:
        """
        Samples delta uniformly in the range [context_lower_bound, context_upper_bound].
        Deltas are a multiplicative scaling factor for each state dimension.
        Args:
            size: number of betas to sample
        Returns:
            betas: tensor of shape [size, z_dimension]
        """

        betas = (
            torch.FloatTensor(size, self._z_dimension)
            .uniform_(self._beta_lower_bound, self._beta_upper_bound)
            .to(self._device)
        )

        return betas

    def infer_z(
        self,
        replay_buffer: MemoryEfficientOfflineReplayBuffer,
        multiplier: int,
        popgym: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Infers z from observations and rewards. Follows page 17 from:
        https://arxiv.org/pdf/2209.14935.pdf. If no rewards are passed, we
        assume we're inferring z from a goal-reaching task, and thus set z
        directly from the goal state.
        Args:
            replay_buffer: replay buffer containing transitions
        Returns:
            z: tensor of shape [z_dimension,]
        """
        multipliers = (multiplier, multiplier)
        zs = {}

        z_inference_chunks = int(self.z_inference_steps / 10000)
        # sample goals/rewards/positions
        for _ in range(z_inference_chunks):
            (
                goals,
                rewards,
                positions,
                actions,
            ) = replay_buffer.sample_task_inference_transitions(
                inference_steps=10000, popgym=popgym
            )

            goals = goals[multipliers]
            rewards = rewards[multipliers]
            positions = positions[multipliers]
            actions = actions[multipliers]

            with torch.no_grad():
                for task, reward in rewards.items():
                    if task not in zs:
                        zs[task] = []

                    if self._recurrent_phi:
                        phi = self.feature_net.forward(goal=goals, action=actions)
                    else:
                        phi = self.feature_net(goal=goals)
                    z = torch.linalg.lstsq(phi, reward).solution
                    z = math.sqrt(self._z_dimension) * torch.nn.functional.normalize(
                        z, dim=0
                    )

                    z = z.squeeze().cpu().numpy()

                    zs[task].append(z)

        for task in zs:
            zs[task] = np.mean(zs[task], axis=0)

        return zs

    def infer_z_from_goal(
        self, observation: np.ndarray, goal_state: torch.Tensor, step: int
    ) -> torch.Tensor:
        """
        Infers z w.r.t. a goal state and the current observation
        Args:
            observation: array of shape [goal_length]
            goal_state: tensor of shape [goal_length]
            step: current step in env
        Returns:
            z: tensor of shape [z_dimension]
        """
        obs = torch.as_tensor(
            observation, dtype=torch.float32, device=self._device
        ).unsqueeze(0)

        if step == 0 and self.frames > 1:
            self._reset_frames(first_observation=obs)

        if self.frames > 1:
            past_frames = (
                torch.stack(list(self.observation_frames), dim=0)
                .squeeze(1)
                .unsqueeze(0)
            )
            frames = torch.cat([past_frames, obs.unsqueeze(1)], dim=1)
            obs = frames.reshape(1, -1)

            # tile the goal state (i.e. assume the goal is to remain at the
            # goal state for self.frames)
            goal_state = goal_state.repeat(self.frames, 1).reshape(1, -1)

        with torch.no_grad():
            obs = self.encoder(obs)
            desired_goal = self.encoder(goal_state)

        with torch.no_grad():
            z_g = self.feature_net(desired_goal)
            z_s = self.feature_net(obs)

        z = z_g - z_s
        z = math.sqrt(self._z_dimension) * torch.nn.functional.normalize(z, dim=1)

        z = z.squeeze().cpu().numpy()

        return z

    def compute_cov(self, goals: torch.Tensor, actions: torch.Tensor) -> None:
        """
        Computes the inverse of the covariance matrix of the features and
        stores internally. This is performed at the beginning of each
        evaluation on goal reaching tasks.
        Args:
            goals: tensor of shape [inference_steps, goal_length]
            actions: tensor of shape [inference_steps, action_length]
        Returns:
            None
        """

        with torch.no_grad():
            phi = self.feature_net(goal=goals, action=actions)

        cov = torch.matmul(phi.T, phi) / phi.shape[0]
        inv_cov = torch.linalg.pinv(cov)

        self.inv_cov = inv_cov

    def predict_q(
        self, observation: torch.Tensor, z: torch.Tensor, action: torch.Tensor
    ):
        """
        Predicts the value of a state-action pair w.r.t. a task. Used as a utility
        function for downstream analysis.
        Args:
            observation: tensor of shape [N, observation_length]
            z: tensor of shape [N, z_dimension]
            action: tensor of shape [N, action_length]
        Returns:
            Qs
        """
        F1, F2 = self.successor_net(observation=observation, z=z, action=action)

        # get Qs from F and z
        Q1 = torch.einsum("sd, sd -> s", F1, z)
        Q2 = torch.einsum("sd, sd -> s", F2, z)
        Q = torch.min(Q1, Q2)

        return Q

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
