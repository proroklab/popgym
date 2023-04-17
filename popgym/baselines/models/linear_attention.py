from typing import List, Tuple

import torch
from torch import nn

from popgym.baselines.models.aggregations import get_aggregator


class Phi(nn.Module):
    def forward(self, x):
        return torch.nn.functional.elu(x) + 1


class LinearAttentionBlock(nn.Module):
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
        self.key = nn.Linear(input_size, hidden_size, bias=False)
        self.query = nn.Linear(input_size, hidden_size, bias=False)
        self.value = nn.Linear(input_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(input_size)
        self.phi = Phi()
        self.S_aggregator = get_aggregator(S_aggregator)()
        self.Z_aggregator = get_aggregator(Z_aggregator)()
        self.feed_forward = feed_forward
        self.residual = residual

        if self.feed_forward:
            self.ff = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True)
            )
        if self.residual:
            self.shortcut = nn.Linear(input_size, hidden_size)

    def forward(
        self, x: torch.Tensor, state: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Input:
            x: [B, T, F]
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

        x = self.norm(x)
        K = self.phi(self.key(x))
        Q = self.phi(self.query(x))
        V = self.value(x)
        S, Z = state
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
        denominator = torch.einsum("bti, btl -> bt", Q, Z).reshape(B, T, 1) + 1e-5
        # output = (Q^T S) / (Q^T Z)
        output = numerator / denominator

        if self.feed_forward:
            output = self.ff(output)

        if self.residual:
            output = output + self.shortcut(x)

        state = [S, Z]

        return output, state
