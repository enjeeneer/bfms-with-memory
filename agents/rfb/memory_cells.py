"""
Module containing memory cells for RFB; heavily informed by
POPGym (Morad et al. (2023)):
https://github.com/proroklab/popgym/tree/master/popgym/baselines/models
"""

import math
import torch
from typing import List, Tuple, Optional


def get_aggregator(name: str) -> torch.nn.Module:
    assert name in [
        "sum",
        "max",
    ], "Invalid aggregator. Must be 'sum' or 'max'"
    return {
        "sum": SumAggregation,
        "max": MaxAggregation,
    }[name]


class Aggregation(torch.nn.Module):
    """Aggregates (x_k ... x_t , s_k) into s_t"""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError()


class SumAggregation(Aggregation):
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        return x.cumsum(dim=1).clamp(-1e20, 1e20) + memory


class MaxAggregation(Aggregation):
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        return torch.maximum(x.cummax(dim=1).values, memory)


class Phi(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.elu(x) + 1


def positional_encoding(
    max_episode_length: int, memory_hidden_dimension: int, device: torch.device
):
    pe = torch.zeros(max_episode_length, memory_hidden_dimension, device=device)
    position = torch.arange(
        0, max_episode_length, dtype=torch.float, device=device
    ).unsqueeze(1)
    # Changed log from 10_000.0 to max_sequence_length, improves
    # accuracy for hard labyrinth
    div_term = torch.exp(
        torch.arange(0, memory_hidden_dimension, 2, device=device).float()
        * (-math.log(max_episode_length) / memory_hidden_dimension)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class LinearAttentionBlock(torch.nn.Module):
    """
    The building block from the Linear Transformers are Secretly RNNs Paper. This is
    a form of linear transformer.

    Inputs:
        input_size: Size of input feature dim
        hidden_size: Size of key/query/value space
        S_aggregator: Which type of aggregation to use for the numerator (S term)
        Z_aggregator: Which type of aggregation to use for the denominator (Z term)
        feed_forward: Whether to apply a perceptron to the output
        residual: Whether to apply a residual connection from input to output
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        S_aggregator: str = "sum",
        Z_aggregator: str = "sum",
        feed_forward=True,
        residual=True,
    ):
        super().__init__()

        self.key = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.query = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.value = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.norm = torch.nn.LayerNorm(input_size)
        self.phi = Phi()
        self.S_aggregator = get_aggregator(S_aggregator)()
        self.Z_aggregator = get_aggregator(Z_aggregator)()
        self.feed_forward = feed_forward
        self.residual = residual

        if self.feed_forward:
            self.ff = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(inplace=True),
            )
        if self.residual:
            self.shortcut = torch.nn.Linear(input_size, hidden_size)

        self._hidden_size = hidden_size

    def forward(
        self,
        history: torch.Tensor,
        previous_hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Input:
            history: [B, T, F]
            state: Tuple[
                [B, 1, D, D],
                [B, 1, D]
            ]
        Output:
            y: [B, T, D]
            state: Tuple[
                [B, 1, D, D],
                [B, 1, D]
            ]
        """

        x = self.norm(history)
        K = self.phi(self.key(x))
        Q = self.phi(self.query(x))
        V = self.value(x)

        # initialize S and Z as zeros if state not passed
        if previous_hidden_state is None:
            S = torch.zeros(
                history.shape[0],
                1,
                self._hidden_size,
                self._hidden_size,
                device=history.device,
            )
            Z = torch.zeros(
                history.shape[0], 1, self._hidden_size, device=history.device
            )
        else:
            S, Z = previous_hidden_state

        B, T, F = K.shape

        # S = sum(K V^T)
        S = self.S_aggregator(
            torch.einsum("bti, btj -> btij", K, V).reshape(B, T, F * F),
            S.reshape(B, 1, F * F),
        ).reshape(B, T, F, F)
        # Z = sum(K)
        Z = self.Z_aggregator(K, Z.reshape(B, 1, F))
        # numerator = Q^T S
        numerator = torch.einsum("bti, btil -> btl", Q, S)
        # denominator = Q^T Z
        denominator = torch.einsum("bti, bti -> bt", Q, Z).reshape(B, T, 1) + 1e-5
        # output = (Q^T S) / (Q^T Z)
        output = numerator / denominator

        if self.feed_forward:
            output = self.ff(output)

        if self.residual:
            output = output + self.shortcut(x)

        state = [S, Z]

        return output, state
