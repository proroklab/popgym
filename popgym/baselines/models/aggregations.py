import math

import torch
from torch import nn


def get_aggregator(name: str) -> nn.Module:
    assert name in [
        "sum",
        "max",
    ], "Invalid aggregator. Must be 'sum' or 'max'"
    return {
        "sum": SumAggregation,
        "max": MaxAggregation,
    }[name]


class Aggregation(nn.Module):
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
