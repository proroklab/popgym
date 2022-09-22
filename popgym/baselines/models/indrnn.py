from typing import Optional, Tuple

import torch
from torch import nn


class IndRNN(nn.Module):
    def __init__(
        self, input_size, hidden_size, activation="relu", clamp=True, max_len=1024
    ):
        super().__init__()
        self.clamp = clamp
        self.clamp_val = 2 ** (1 / max_len)
        self.input0 = nn.Linear(input_size, hidden_size)
        self.weight0 = nn.Parameter(torch.empty(hidden_size))
        self.bias0 = nn.Parameter(torch.empty(hidden_size))

        self.input1 = nn.Linear(hidden_size, hidden_size)
        self.weight1 = nn.Parameter(torch.empty(hidden_size))
        self.bias1 = nn.Parameter(torch.empty(hidden_size))

        self.output = nn.Linear(hidden_size, hidden_size)

        torch.nn.init.uniform_(self.weight0, 0, 1)
        torch.nn.init.uniform_(self.weight1, 0, 1)
        torch.nn.init.uniform_(self.bias0, 0, 1)
        torch.nn.init.uniform_(self.bias1, 0, 1)

    def clamp_weights(self) -> None:
        self.weight0.data.clamp_(-self.clamp_val, self.clamp_val)
        self.weight1.data.clamp_(-self.clamp_val, self.clamp_val)

    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input:
            x: [B, T, F]
            state: [B, 2, F]
        output:
            y: [B, T, D]
            state: [B, 2, D]
        """
        if state is None:
            state = torch.zeros(x.shape[0], 2, *self.weight0.shape, device=x.device)

        state0, state1 = state.chunk(2, dim=1)
        B, T, _ = x.shape
        if self.clamp:
            self.clamp_weights()
        x = self.input0(x)
        outs = []
        for t in range(T):
            state0 = torch.relu(x[:, t, None] + state0 * self.weight0 + self.bias0)
            state1 = torch.relu(
                self.input1(state0) + state1 * self.weight1 + self.bias1
            )
            outs.append(state1)

        outs = torch.cat(outs, dim=1)
        return outs, torch.stack([state0[:, -1], state1[:, -1]], dim=1)
