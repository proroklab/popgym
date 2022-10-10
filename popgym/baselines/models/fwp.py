import torch
from torch import nn

from popgym.baselines.models.aggregations import get_aggregator


class FWPBlock(nn.Module):
    """
    The building block in the fast weight transformers paper. This is
    a form of linear transformer.

    Inputs:
        input_size: Size of input feature dim
        hidden_size: Size of key/query/value space
        aggregator: Which type of aggregation to use
        sum_normalization: Whether to use the sum normalization described
            in the paper
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        aggregator="sum",
        sum_normalization=True,
    ):
        super().__init__()
        self.key = nn.Linear(input_size, hidden_size, bias=False)
        self.query = nn.Linear(input_size, hidden_size, bias=False)
        self.value = nn.Linear(input_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(input_size)
        self.sum_normalization = sum_normalization
        self.aggregator = get_aggregator(aggregator)(
            max_len=1024, d_model=hidden_size**2
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            x: [B, T, F]
            state: [B, 1, F]
        Outputs:
            y: [B, T, D]
            state: [B, T, F]
        """
        x = self.norm(x)
        K = self.key(x).relu()
        Q = self.query(x).relu()
        V = self.value(x)
        if self.sum_normalization:
            K = K / (1e-5 + K.sum(dim=-1, keepdim=True))
            Q = Q / (1e-5 + Q.sum(dim=-1, keepdim=True))

        kv = torch.einsum("bti, btj -> btij", V, K)
        shape = kv.shape
        state = self.aggregator(kv.flatten(-2), state.flatten(-2)).reshape(shape)

        y = torch.einsum("btij, bti -> btj", state, Q)
        return y, state
