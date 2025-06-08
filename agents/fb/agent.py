"""Module defining the Forward-Backward Agent."""

import math
from pathlib import Path
from typing import Tuple, Dict, Optional, Union
from collections import deque

import torch
import numpy as np

from agents.fb.models import ForwardBackwardRepresentation, ActorModel
from agents.base import (
    AbstractAgent,
    Batch,
    AbstractGaussianActor,
    make_aug_encoder,
    MemoryEfficientOfflineReplayBuffer,
)
from agents.utils import schedule


class FB(AbstractAgent):
    """
    Forward-Backward Agent
    """

    def __init__(
        self,
        observation_dims: Union[int, Tuple[int, int, int]],
        observation_type: str,
        action_length: int,
        goal_dimension: int,
        preprocessed_dimension: int,
        postprocessed_dimension: int,
        z_dimension: int,
        forward_hidden_dimension: int,
        forward_hidden_layers: int,
        backward_hidden_dimension: int,
        backward_hidden_layers: int,
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        forward_activation: str,
        backward_activation: str,
        actor_activation: str,
        actor_learning_rate: float,
        critic_learning_rate: float,
        learning_rate_coefficient: float,
        orthonormalisation_coefficient: float,
        discount: float,
        batch_size: int,
        z_mix_ratio: float,
        gaussian_actor: bool,
        std_dev_clip: float,
        std_dev_schedule: str,
        tau: float,
        device: torch.device,
        name: str,
        exploration_epsilon: float,
        frames: int,
        boltzmann_temperature: float,
        z_inference_steps: int,
        action_in_B: bool = False,
        discrete_actions: bool = False,
        layernorms: bool = True,
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

        self.FB = ForwardBackwardRepresentation(
            observation_length=observation_length,
            action_length=action_length,
            goal_dimension=goal_dimension,
            preprocessed_dimension=preprocessed_dimension,
            postprocessed_dimension=postprocessed_dimension,
            z_dimension=z_dimension,
            forward_hidden_dimension=forward_hidden_dimension,
            forward_hidden_layers=forward_hidden_layers,
            backward_hidden_dimension=backward_hidden_dimension,
            backward_hidden_layers=backward_hidden_layers,
            forward_activation=forward_activation,
            backward_activation=backward_activation,
            orthonormalisation_coefficient=orthonormalisation_coefficient,
            discount=discount,
            device=device,
            discrete_actions=discrete_actions,
            layernorms=layernorms,
            include_action_in_B=action_in_B,
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
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=actor_learning_rate
            )

        # load weights into target networks
        self.FB.forward_representation_target.load_state_dict(
            self.FB.forward_representation.state_dict()
        )
        self.FB.backward_representation_target.load_state_dict(
            self.FB.backward_representation.state_dict()
        )
        if observation_type == "pixels":
            self.image_encoder_optimizer = torch.optim.Adam(
                self.image_encoder.parameters(), lr=critic_learning_rate
            )
        else:
            self.image_encoder_optimizer = None

        self._device = device
        self.batch_size = batch_size
        self._z_mix_ratio = z_mix_ratio
        self._tau = tau
        self._action_in_B = action_in_B
        self._z_dimension = z_dimension
        self._exploration_epsilon = exploration_epsilon
        self._discrete_actions = discrete_actions
        self._boltzmann_temperature = boltzmann_temperature
        self.z_inference_steps = z_inference_steps
        self.std_dev_schedule = std_dev_schedule
        self.frames = frames

    def _reset_frames(self, first_observation: torch.Tensor) -> None:
        """
        Initialises memory with duplicates of the first observation.
        """

        self.observation_frames = deque(
            [first_observation] * (self.frames - 1),
            maxlen=(self.frames - 1),
        )

    @torch.no_grad()
    def act(
        self,
        observation: np.ndarray,
        task: np.array,
        step: int,
        sample: bool = False,
    ) -> Tuple[np.array, float]:
        """
        Used at test time to perform zero-shot rollouts.
        Takes observation array from environment, encodes, and selects
        action from actor.
        Args:
            observation: observation array of shape [observation_length]
            task: task array of shape [z_dimension]
            step: current step in env
            sample: whether to sample action from actor distribution
        Returns:
            action: action array of shape [action_length]
            std_dev: current actor standard deviation
        """

        observation = torch.as_tensor(
            observation, dtype=torch.float32, device=self._device
        ).unsqueeze(0)

        # only do internal frame stack for states. For pixels the
        # frame stacking is handled by the env
        if step == 0 and self.frames > 1 and self.observation_type == "states":
            self._reset_frames(first_observation=observation)

        if self.frames > 1 and self.observation_type == "states":
            past_frames = (
                torch.stack(list(self.observation_frames), dim=0)
                .squeeze(1)
                .unsqueeze(0)
            )
            frames = torch.cat([past_frames, observation.unsqueeze(1)], dim=1)
            encoder_input = frames.reshape(1, -1)
        else:
            encoder_input = observation

        # flatten stacked frames if pixel based observations
        # if self.observation_type == "pixels":
        #     observation = observation.reshape(-1, 64, 64).unsqueeze(0)

        h = self.image_encoder(encoder_input)
        z = torch.as_tensor(task, dtype=torch.float32, device=self._device).unsqueeze(0)

        # if discrete actions, get argmax of Q values
        if self._discrete_actions:
            F1, F2 = self.FB.forward_representation.forward(observation=h, z=z)
            Q1, Q2 = [torch.einsum("sda, sd -> sa", Fi, z) for Fi in [F1, F2]]
            Q = torch.min(Q1, Q2)
            action = Q.max(1)[1].unsqueeze(0)

            if sample:
                action = (
                    torch.randint_like(action, self.action_length)
                    if np.random.rand() < self._exploration_epsilon
                    else action
                )

            std_dev = 0.0

        # if continuous actions, sample from actor
        else:
            # get action from actor
            std_dev = schedule(self.std_dev_schedule, step)
            action, _ = self.actor(observation=h, z=z, std=std_dev, sample=sample)

        # add frame to memory
        if self.frames > 1 and self.observation_type == "states":
            self.observation_frames.append(observation)

        action = action.detach().cpu().numpy()

        return np.squeeze(action, 0), std_dev

    def _aug_and_encode(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.augmentation(obs)
        return self.image_encoder(obs)

    def update(self, batch: Batch, step: int) -> Dict[str, float]:
        """
        Updates agent's networks given a batch_size sample from the replay buffer.
        Args:
            batch: memory buffer containing transitions
            step: no. of steps taken in the environment
        Returns:
            metrics: dictionary of metrics for logging
        """

        # augment and encode
        observations = self._aug_and_encode(batch.observations)
        next_observations = self._aug_and_encode(batch.next_observations)
        backward_input = self._aug_and_encode(batch.goals)
        next_goals = self._aug_and_encode(batch.next_goals)
        next_goal_actions = self._aug_and_encode(batch.next_goal_action_histories)
        next_observations = next_observations.detach()
        backward_input = backward_input.detach()

        # sample zs
        zs = self._sample_z(size=self.batch_size)
        perm = torch.randperm(self.batch_size)
        backward_goals = backward_input[perm]
        if self._action_in_B:
            backward_actions = next_goal_actions[perm]

        mix_indices = np.where(np.random.rand(self.batch_size) < self._z_mix_ratio)[0]
        with torch.no_grad():
            mix_zs = self.FB.backward_representation.forward(
                goal=backward_goals[mix_indices],
                action=backward_actions[mix_indices] if self._action_in_B else None,
            ).detach()

        zs[mix_indices] = mix_zs

        # update forward and backward models
        fb_metrics = self.update_fb(
            observations=observations,
            next_observations=next_observations,
            next_goals=next_goals,
            next_goal_actions=next_goal_actions,
            actions=batch.actions.to(torch.int64)
            if self._discrete_actions
            else batch.actions,
            discounts=batch.discounts,
            zs=zs,
            step=step,
        )

        actor_metrics = {}
        if not self._discrete_actions:

            # update actor
            actor_metrics = self.update_actor(
                observation=observations.detach(), z=zs, step=step
            )

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
        }

        return metrics

    def update_fb(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        next_goals: torch.Tensor,
        next_goal_actions: torch.Tensor,
        discounts: torch.Tensor,
        zs: torch.Tensor,
        step: int,
    ) -> Dict[str, float]:
        """
        Updates the forward-backward representation network.
        Args:
            observations: observation tensor of shape [batch_size, observation_length]
            actions: action tensor of shape [batch_size, action_length]
            next_observations: next observation tensor of
                                shape [batch_size, observation_length]
            next_goals: next goal tensor of shape [batch_size, goal_length]
            next_goal_actions: next goal action tensor of shape
                                    [batch_size, action_length]
            discounts: discount tensor of shape [batch_size, 1]
            zs: policy tensor of shape [batch_size, z_dimension]
            step: current training step
        Returns:
            metrics: dictionary of metrics for logging
        """

        if self._discrete_actions:
            assert actions.shape == (self.batch_size, 1), (
                f"Batch of discrete actions in dataset have shape {actions.shape}, but"
                f"this implementation expects shape (batch_size, 1)."
            )

        total_loss, metrics, _, _, _, _, _, _, _, _ = self._update_fb_inner(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            next_goals=next_goals,
            next_goal_actions=next_goal_actions,
            discounts=discounts,
            zs=zs,
            step=step,
        )

        if self.image_encoder_optimizer is not None:
            self.image_encoder_optimizer.zero_grad(set_to_none=True)
        self.FB_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        # for param in self.FB.parameters():
        #     if param.grad is not None:
        #         param.grad.data.clamp_(-1, 1)
        self.FB_optimizer.step()
        if self.image_encoder_optimizer is not None:
            self.image_encoder_optimizer.step()

        return metrics

    def _update_fb_inner(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        next_goals: torch.Tensor,
        next_goal_actions: torch.Tensor,
        discounts: torch.Tensor,
        zs: torch.Tensor,
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
            observations: observation tensor of shape [batch_size, observation_length]
            actions: action tensor of shape [batch_size, action_length]
            next_observations: next observation tensor of
                                shape [batch_size, observation_length]
            next_goals: next goal tensor of shape [batch_size, goal_length]
            next_goal_actions: next goal action tensor of shape
                                                    [batch_size, action_length]
            discounts: discount tensor of shape [batch_size, 1]
            zs: policy tensor of shape [batch_size, z_dimension]
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

        with torch.no_grad():

            if self._discrete_actions:
                # compute greedy action
                target_F1, target_F2 = self.FB.forward_representation_target(
                    observation=next_observations, z=zs
                )
                next_Q1, next_Q2 = [
                    torch.einsum("sda, sd -> sa", Fi, zs)
                    for Fi in [target_F1, target_F2]
                ]
                next_Q = torch.min(next_Q1, next_Q2)
                pi = torch.nn.functional.softmax(
                    next_Q / self._boltzmann_temperature, dim=-1
                )
                target_F1, target_F2 = [
                    torch.einsum("sa, sda -> sd", pi, Fi)
                    for Fi in [target_F1, target_F2]
                ]  # batch x z_dim
                actor_std_dev = 0.0

            else:
                # sample actions from actor
                actor_std_dev = schedule(self.std_dev_schedule, step)
                next_actions, _ = self.actor(
                    next_observations, zs, actor_std_dev, sample=True
                )

                target_F1, target_F2 = self.FB.forward_representation_target(
                    observation=next_observations, z=zs, action=next_actions
                )

            target_B = self.FB.backward_representation_target.forward(
                goal=next_goals,
                action=next_goal_actions if self._action_in_B else None,
            )
            target_M1 = torch.einsum(
                "sd, td -> st", target_F1, target_B
            )  # [batch_size, batch_size]
            target_M2 = torch.einsum(
                "sd, td -> st", target_F2, target_B
            )  # [batch_size, batch_size]
            target_M = torch.min(target_M1, target_M2)

        # --- Forward-backward representation loss ---
        if self._discrete_actions:
            idxs = actions.repeat(1, self._z_dimension)[:, :, None]
            F1, F2 = [
                Fi.gather(-1, idxs).squeeze()
                for Fi in self.FB.forward_representation(observation=observations, z=zs)
            ]
        else:
            F1, F2 = self.FB.forward_representation.forward(
                observation=observations, action=actions, z=zs
            )

        B_next = self.FB.backward_representation.forward(
            goal=next_goals,
            action=next_goal_actions if self._action_in_B else None,
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

    def update_actor(
        self, observation: torch.Tensor, z: torch.Tensor, step: int
    ) -> Dict[str, float]:
        """
        Computes the actor loss.
        Args:
            observation: tensor of shape [batch_size, observation_length]
            z: tensor of shape [batch_size, z_dimension]
            step: current training step
        Returns:
            metrics: dictionary of metrics for logging
        """
        std = schedule(self.std_dev_schedule, step)

        action, action_dist = self.actor(
            observation=observation,
            z=z,
            std=std,
            sample=True,
        )

        F1, F2 = self.FB.forward_representation(
            observation=observation, z=z, action=action
        )

        # get Qs from F and z
        Q1 = torch.einsum("sd, sd -> s", F1, z)
        Q2 = torch.einsum("sd, sd -> s", F2, z)
        Q = torch.min(Q1, Q2)

        # update actor towards action that maximise Q (minimise -Q)
        actor_loss = -Q

        if (
            type(self.actor.actor)  # pylint: disable=unidiomatic-typecheck
            == AbstractGaussianActor
        ):
            # add an entropy regularisation term
            log_prob = action_dist.log_prob(action).sum(-1)
            actor_loss += 0.1 * log_prob  # NOTE: currently hand-coded weight!
            mean_log_prob = log_prob.mean().item()
        else:
            mean_log_prob = 0.0

        actor_loss = actor_loss.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        # for param in self.actor.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()

        metrics = {
            "train/actor_loss": actor_loss.item(),
            "train/actor_Q": Q.mean().item(),
            "train/actor_log_prob": mean_log_prob,
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

                with torch.no_grad():
                    z = self.FB.backward_representation.forward(
                        goals,
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
                    _,
                    actions,
                ) = replay_buffer.sample_task_inference_transitions(
                    inference_steps=10000,
                    popgym=popgym,
                )

                goals = goals[multipliers]
                rewards = rewards[multipliers]
                actions = actions[multipliers]

                with torch.no_grad():
                    goals = self.image_encoder(goals)

                with torch.no_grad():
                    for task, reward in rewards.items():
                        if task not in zs:
                            zs[task] = []
                        z = self.FB.backward_representation.forward(
                            goal=goals, action=actions if self._action_in_B else None
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
        F1, F2 = self.FB.forward_representation.forward(
            observation=observation, z=z, action=action
        )

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
