import math

import torch
from torch import nn


def positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    # Changed log from 10_000.0 to max_len, improves accuracy for hard labyrinth
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(max_len) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


@torch.jit.script
def _psi(ab: torch.Tensor, c: torch.Tensor, t_minus_i: torch.Tensor) -> torch.Tensor:
    # ab: [1, T, F ... F]
    # c: [1, T, 1 ... 1]
    # t_minus_i: [1, T, 1 ... 1]
    return torch.exp(c * ab * t_minus_i)


@torch.jit.script
def _compute_state(
    ab: torch.Tensor,
    c: torch.Tensor,
    x: torch.Tensor,
    n_to_zero: torch.Tensor,
    one_to_np1: torch.Tensor,
    T: int,
    prev_state: torch.Tensor,
):
    state = torch.cumsum(x * _psi(ab, c, n_to_zero[:T]), dim=1) * _psi(
        ab, c, -n_to_zero[:T]
    )
    # Add to previous output
    # Shift the previous inputs by 1 to T
    # because they are now 1 to T timesteps older with the
    # addition of T new timesteps present in x
    psi_shift = _psi(ab, c, one_to_np1[:T])
    shifted_prev_state = prev_state * psi_shift
    output = state + shifted_prev_state
    return output


class PhaserEncoding(nn.Module):
    def __init__(
        self,
        max_len: int,
        d_model: int,
        num_dims: int,
        max_period: int = 1024,
        dtype: torch.dtype = torch.double,
        decay_bias: float = -0.5,
        freq_scale: float = 1.0,
        fudge_factor: float = 0.025,
        soft_clamp: bool = False,
    ):
        """A phaser-encoded aggregation operator
        Inputs:
            max_len: Maximum length of the batch in timesteps. Note this
                is not the episode length, but rather the batch length.
            d_model: Feature dimension of the model
            num_dims: Number of feature dimensions of the model, each should be
                of size d_model
            max_period: This determines the initial maximum sinusoidal period. It should
                be set to the maximum episode length if known. Until the parameters
                change, the model will be unable to differentiate relative time
                differences greater than max_period.
            dtype: Whether to use floats or doubles. Note doubles enables
                significantly more representational power for a little
                extra compute
            decay_bias: How much bias to exponential decay. Higher -> biased
                towards shorter-term memory. This should roughly be between 0 and 5
                since this value goes through a sigmoid activation.
            freq_scale: How much to bias the sinusoidal part of the encoding.
                Lower -> lower frequencies, biased towards longer-term memory
                    and being unable to tell the difference between x_t and x_{t+1}
                Higher -> higher frequencies, biased towards quick reactions/control
                    and more likely to differentiate between x_t and x_{t+1}
                This should be a positive number.
            fudge_factor: A small positive number to prevent overflows
        """
        super().__init__()
        self.num_dims = num_dims
        self.d_model = d_model
        # Set the upper bound to zero because our prior knowledge
        # dictates experiences from the past never become stronger,
        # only fade. Also, this could cause explosions as we magnify
        # values at each timestep
        # To prevent overflows, ensure exp(limit * max_len) < {float,double}
        # limit * max_len < log({float,double})
        # limit == log({float,double}) / max_len - fudge_factor
        assert dtype in [torch.float, torch.double]
        self.dtype = dtype
        dtype_max = torch.finfo(dtype).max
        self.limit = math.log(dtype_max) / max_len - fudge_factor
        self.soft_clamp = soft_clamp

        # [1, 1 | 1, ... 1, F]
        # [B, T | F, ... F, F]
        param_shape = [1, 1] + [self.d_model] * num_dims
        a = torch.linspace(
            0 - fudge_factor, -self.limit + fudge_factor, self.d_model**num_dims
        ).reshape(param_shape)
        b = (
            2
            * torch.pi
            / torch.linspace(max_period, torch.pi, self.d_model**num_dims).reshape(
                param_shape
            )
        )
        # b = b.flatten()[torch.randperm(b.numel())].reshape(b.shape)
        self.c = nn.Parameter(torch.tensor(decay_bias))
        self.d = nn.Parameter(torch.tensor(freq_scale))
        self.ab = nn.Parameter(torch.complex(a, b))

        torch.nn.init.normal_(self.ab.real, mean=0, std=0.01)
        self.ab.real.data.clamp_(-self.limit + fudge_factor, 0 - fudge_factor)

        n_to_zero = torch.arange(max_len, dtype=dtype).flip(0)
        self.register_buffer("n_to_zero", n_to_zero)
        one_to_np1 = torch.arange(1, max_len + 1, dtype=dtype)
        self.register_buffer("one_to_np1", one_to_np1)

    def psi(self, t_minus_i):
        assert t_minus_i.dim() == 1

        T = t_minus_i.shape[0]
        F = [self.d_model] * self.num_dims
        broadcast = [1] * len(F)
        # Compute for all filters/fourier series terms
        # e^(t * (a + bi))
        if self.soft_clamp:
            real = -self.limit * (self.c + self.ab.real).sigmoid()
        else:
            real = (self.c + self.ab.real).clamp(-self.limit, 0)
        imag = self.d * self.ab.imag
        return torch.exp(
            torch.complex(real, imag) * t_minus_i.reshape(1, T, *broadcast)
        )
        # return _psi(self.ab, self.c, t_minus_i.reshape(1, T, *broadcast))

    def recurrent_update2(self, x, state):
        B, T, *F = x.shape

        prev_state = torch.view_as_complex(state)
        ones = torch.tensor([-1.0, 1.0], dtype=torch.complex128, device=x.device)

        if self.training:
            psi_minus_one, psi_one = self.psi(ones).unbind(1)
        else:
            psi_minus_one, psi_one = self.psi(ones).unbind(1)
        psi_n_to_zero = psi_one.expand(1, T, -1, -1, -1).cumprod(1).flip(1)
        minus_psi_n_to_zero = psi_minus_one.expand(1, T, -1, -1, -1).cumprod(1).flip(1)

        # Convert back to float for performance
        state = torch.cumsum(x * psi_n_to_zero, dim=1) * minus_psi_n_to_zero

        shift = psi_n_to_zero.flip(1) * psi_one
        shifted_prev_state = prev_state * shift
        output = state + shifted_prev_state
        return output.to(torch.complex64)

    def recurrent_update(self, x, state):
        B, T, *F = x.shape
        prev_state = torch.view_as_complex(state)

        state = torch.cumsum(x * self.psi(self.n_to_zero[:T]), dim=1) * self.psi(
            -self.n_to_zero[:T]
        )
        # Add to previous output
        # Shift the previous inputs by 1 to T
        # because they are now 1 to T timesteps older with the
        # addition of T new timesteps present in x
        psi_shift = self.psi(self.one_to_np1[:T])
        shifted_prev_state = prev_state * psi_shift
        output = state + shifted_prev_state
        return output.to(torch.complex64)

    def forward(self, x, state):
        return self.recurrent_update(x, state)


class ScaledPositionalEncoding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        self.a = nn.Parameter(torch.rand(embedding_dim))
        num_zero_weights = round(0.75 * embedding_dim)
        zero_idx = torch.randperm(embedding_dim)[:num_zero_weights]
        self.a.data[zero_idx] = 0

    def forward(self, idx):
        return super().forward(idx) * self.a


class SimpleScaledEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        v = torch.zeros(d_model)
        v[-4:] = 1.0 / max_len
        self.register_buffer("v", v)

    def forward(self, idx):
        return idx.reshape(-1, 1) * self.v


class Time2Vec(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.net = nn.Linear(1, d_model)
        # self.emb = nn.Embedding(max_len, 2 * d_model)

    def forward(self, idx):
        # [B * T, 1]
        idx = idx.reshape(-1, 1)
        emb = self.net(idx.float()).reshape(-1, self.d_model)
        zero, rest = emb.split([1, self.d_model - 1], dim=-1)
        return torch.cat([zero, rest.sin()], dim=-1)


class Base2Embedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        mask = 2 ** torch.arange(d_model)
        # Overflow, zero these for sanity
        # seq_len <= 65536
        mask[16:] = mask[16]
        assert max_len < 2**16 - 1
        self.register_buffer("mask", mask)

    def forward(self, idx):
        return idx.unsqueeze(-1).bitwise_and(self.mask).ne(0).float()


class PerceptronEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.d_model = d_model
        self.lin = nn.Linear(1, d_model, bias=False)

    def forward(self, idx):
        idx = idx.float()
        return self.lin(idx.reshape(-1, 1)).sin()


class SoftExponentialEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.d_model = d_model
        self.a = nn.Parameter([-1.0])
        self.lin = nn.Linear(1, d_model, bias=False)

    def forward(self, idx):
        idx = idx.float()
        self.a.data = self.a.data.clamp_(-1.0, 1e-4)
        (torch.exp(self.a * idx) - 1) / self.a + self.a
        return self.lin(idx.reshape(-1, 1)).sin()
