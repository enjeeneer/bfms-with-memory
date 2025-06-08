"""Feature embedding model for the SF agent."""
import torch
from typing import Dict, Tuple
from agents.base import MLPEncoder


class RecurrentPhi(torch.nn.Module):
    """
    Embeds states in feature space.
    """

    def __init__(
        self,
        goal_dimension: int,
        action_length: int,
        z_dimension: int,
        hidden_dimension: int,
        device: torch.device,
        history_length: int,
        memory_type: str,
        num_encoder_layers: int,
    ):
        self._z_dimension = z_dimension
        self.device = device
        super().__init__()

        # pre-processors
        if memory_type == "mlp":
            # frame-stacking
            goal_action_input_dimension = (
                goal_dimension + action_length
            ) * history_length
        else:
            goal_action_input_dimension = goal_dimension + action_length

        if memory_type == "mlp":
            self.feature_encoder = MLPEncoder(
                raw_input_dimension=goal_action_input_dimension,
                preprocessed_dimension=hidden_dimension,
                postprocessed_dimension=z_dimension,
                device=device,
                layers=num_encoder_layers,
            )

        else:
            raise ValueError(f"Invalid memory type: {memory_type}")

    def forward(self, goal: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Embeds observation in z space and L2 normalizes.
        Args:
            goal: tensor of shape [batch_dim, goal_dim]
            action: tensor of shape [batch_dim, action_dim]
        Returns:
            z: embedded feature tensor of shape [batch_dim, z_dimension]
        """

        goal_action_embedding, _ = self.feature_encoder(
            torch.concat([goal, action], dim=-1),
        )

        # L2 normalize then scale to radius sqrt(z_dimension)
        z = torch.sqrt(
            torch.tensor(self._z_dimension, dtype=torch.int, device=self.device)
        ) * torch.nn.functional.normalize(goal_action_embedding, dim=1)

        return z


class RecurrentHILPFeatures(torch.nn.Module):
    """Recurrent feature embedding learned with HILP loss."""

    def __init__(
        self,
        goal_dimension: int,
        action_length: int,
        z_dimension: int,
        hidden_dimension: int,
        history_length: int,
        memory_type: str,
        num_encoder_layers: int,
        device: torch.device,
        discount: float,
        iql_expectile: float,
    ):
        super().__init__()

        phi_config = [
            goal_dimension,
            action_length,
            z_dimension,
            hidden_dimension,
            device,
            history_length,
            memory_type,
            num_encoder_layers,
        ]

        self.phi1 = RecurrentPhi(*phi_config)
        self.phi2 = RecurrentPhi(*phi_config)
        self.target_phi1 = RecurrentPhi(*phi_config)
        self.target_phi2 = RecurrentPhi(*phi_config)
        self.target_phi1.load_state_dict(self.phi1.state_dict())
        self.target_phi2.load_state_dict(self.phi2.state_dict())

        # Running mean and std for normalizing phi
        self.register_buffer("running_mean", torch.zeros(z_dimension, device=device))
        self.register_buffer("running_std", torch.ones(z_dimension, device=device))

        self.discount = discount
        self.iql_expectile = iql_expectile

    def forward(self, goal: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Taken an observation and returns the feature embedding.
        Args:
            goal: tensor of shape [batch_dim, history_length, goal_dim]
            action: tensor of shape [batch_dim, history_length, action_dim]
        Returns:
            phi: embedded feature tensor of shape [batch_dim, z_dimension]
        """

        phi = self.phi1.forward(goal=goal, action=action)
        phi = phi - self.running_mean

        return phi

    def get_loss(
        self,
        goal_histories: torch.Tensor,
        action_histories: torch.Tensor,
        next_goal_histories: torch.Tensor,
        next_action_histories: torch.Tensor,
        future_goal_histories: torch.Tensor,
        future_action_histories: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes loss for HILP features update.
        Args:
            goal_histories: tensor of shape [batch_dim, history_length, goal_dim]
            action_histories: tensor of shape [batch_dim, history_length, action_dim]
            next_goal_histories: tensor of shape [batch_dim, history_length, goal_dim]
            next_action_histories: tensor of shape
                                            [batch_dim, history_length, action_dim]
            future_goal_histories: tensor of shape
                                            [batch_dim, history_length, goal_dim]
            future_action_histories: tensor of shape
                                            [batch_dim, history_length, action_dim]
        Returns:
            loss: loss for HILP update
            metrics: dictionary of metrics for logging
        """

        # index last observation in history for rewards
        rewards = (
            torch.linalg.norm(
                goal_histories[:, -1, :] - future_goal_histories[:, -1, :], dim=-1
            )
            < 1e-6
        ).float()
        masks = 1.0 - rewards
        rewards = rewards - 1.0

        next_v1, next_v2 = self._value(
            goals=next_goal_histories,
            actions=next_action_histories,
            future_goals=goal_histories,
            future_actions=action_histories,
            is_target=True,
        )
        next_v = torch.minimum(next_v1, next_v2)
        q = rewards + self.discount * masks * next_v

        v1_t, v2_t = self._value(
            goals=goal_histories,
            actions=action_histories,
            future_goals=future_goal_histories,
            future_actions=future_action_histories,
            is_target=True,
        )
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = rewards + self.discount * masks * next_v1
        q2 = rewards + self.discount * masks * next_v2
        v1, v2 = self._value(
            goals=goal_histories,
            actions=action_histories,
            future_goals=future_goal_histories,
            future_actions=future_action_histories,
            is_target=False,
        )
        v = (v1 + v2) / 2

        value_loss1 = self._expectile_loss(adv, q1 - v1, self.iql_expectile).mean()
        value_loss2 = self._expectile_loss(adv, q2 - v2, self.iql_expectile).mean()
        value_loss = value_loss1 + value_loss2

        self._soft_update_params(
            network=self.phi1, target_network=self.target_phi1, tau=0.005
        )
        self._soft_update_params(
            network=self.phi2, target_network=self.target_phi2, tau=0.005
        )

        with torch.no_grad():
            phi1 = self.phi1.forward(goal=goal_histories, action=action_histories)
            self.running_mean = 0.995 * self.running_mean + 0.005 * phi1.mean(dim=0)
            self.running_std = 0.995 * self.running_std + 0.005 * phi1.std(dim=0)

        return value_loss, {
            "train/value_loss": value_loss,
            "train/v_mean": v.mean(),
            "train/v_max": v.max(),
            "train/v_min": v.min(),
            "train/abs_adv_mean": torch.abs(adv).mean(),
            "train/adv_mean": adv.mean(),
            "train/adv_max": adv.max(),
            "train/adv_min": adv.min(),
            "train/accept_prob": (adv >= 0).float().mean(),
        }

    def _value(
        self,
        goals: torch.Tensor,
        actions: torch.Tensor,
        future_goals: torch.Tensor,
        future_actions: torch.Tensor,
        is_target: bool = False,
    ):
        """
        Computes the value of a state w.r.t. to a goal state. The value
        is the negative L2 distance in feature space.
        Args:
            goals: tensor of shape [batch_dim, history_length, goal_dim]
            actions: tensor of shape [batch_dim, history_length, action_dim]
            future_goals: tensor of shape [batch_dim, history_length, goal_dim]
            is_target: whether to use target network
        Returns:
            v1: value of state w.r.t. goal from phi1
            v2: value of state w.r.t. goal from phi2
        """
        if is_target:
            phi1 = self.target_phi1
            phi2 = self.target_phi2
        else:
            phi1 = self.phi1
            phi2 = self.phi2

        phi1_s = phi1.forward(goal=goals, action=actions)
        phi1_g = phi1.forward(goal=future_goals, action=future_actions)

        phi2_s = phi2.forward(goal=goals, action=actions)
        phi2_g = phi2.forward(goal=future_goals, action=future_actions)

        squared_dist1 = ((phi1_s - phi1_g) ** 2).sum(dim=-1)
        v1 = -torch.sqrt(torch.clamp(squared_dist1, min=1e-6))
        squared_dist2 = ((phi2_s - phi2_g) ** 2).sum(dim=-1)
        v2 = -torch.sqrt(torch.clamp(squared_dist2, min=1e-6))

        if is_target:
            v1 = v1.detach()
            v2 = v2.detach()

        return v1, v2

    @staticmethod
    def _expectile_loss(adv, diff, expectile):
        """
        Computes the expectile loss for IQL update.
        Args:
            adv: advantage
            diff: difference between Q and V
            expectile: expectile parameter
        Returns:
            loss: expectile loss
        """
        weight = torch.where(adv >= 0, expectile, (1 - expectile))

        return weight * (diff**2)

    @staticmethod
    def _soft_update_params(
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
