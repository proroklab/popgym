import math
from typing import List, Tuple

import gymnasium as gym
import torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn

from popgym.baselines.models.linear_attention import LinearAttentionBlock
from popgym.baselines.ray_models.base_model import BaseModel


class LinearAttention(BaseModel):
    r"""The Fast Autoregressive Transformer (FART, lol) from

    .. code-block:: text

        @inproceedings{katharopoulos_transformers_2020,
            title = {
                Transformers are RNNs: Fast Autoregressive
                Transformers with Linear Attention
            },
            shorttitle = {Transformers are {RNNs}},
            url = {https://proceedings.mlr.press/v119/katharopoulos20a.html},
            language = {en},
            urldate = {2022-09-21},
            booktitle = {
                Proceedings of the 37th {International}
                {Conference} on {Machine} {Learning}
            },
            publisher = {PMLR},
            author = {
                Katharopoulos, Angelos and Vyas, Apoorv and Pappas,
                Nikolaos and Fleuret, François
            },
            month = nov,
            year = {2020},
            note = {ISSN: 2640-3498},
            pages = {5156--5165},
        }
    """

    MODEL_CONFIG = {
        # Which positional embedding to use
        "embedding": "sine",
        # Which aggregation operator to use for the S term.
        # The paper only uses sum, but this can be sum or max
        "S_aggregator": "sum",
        # Same as S aggregator, except for the Z (denominator) term
        "Z_aggregator": "sum",
    }

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **custom_model_kwargs,
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.h = round(math.sqrt(self.cfg["hidden_size"]) - 1)
        self.core = LinearAttentionBlock(
            input_size=self.cfg["preprocessor_output_size"],
            hidden_size=self.h,
            S_aggregator=self.cfg["S_aggregator"],
            Z_aggregator=self.cfg["Z_aggregator"],
        )
        self.unmap = nn.Linear(self.h, self.cfg["hidden_size"])

    def initial_state(self) -> List[TensorType]:
        return [
            torch.zeros(1, self.h, self.h),
            torch.zeros(1, self.h),
        ]

    def forward_memory(
        self,
        z: TensorType,
        state: List[TensorType],
        t_starts: TensorType,
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        B, T, _ = z.shape
        z, state = self.core(z, state)
        h = z.shape[-1]
        z = self.unmap(z)
        S, Z = state
        return z, [
            S[:, -1].reshape(B, 1, h, h),
            Z[:, -1].reshape(B, 1, h),
        ]


class DeepLinearAttention(LinearAttention):
    """A multi-layer version of linear attention."""
    MODEL_CONFIG = {
        # Which positional embedding to use
        "embedding": None,
        # Which aggregation operator to use for the S term.
        # The paper only uses sum, but this can be sum or max
        "S_aggregator": "sum",
        # Same as S aggregator, except for the Z (denominator) term
        "Z_aggregator": "sum",
        "num_layers": 4,
    }

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **custom_model_kwargs,
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        assert self.cfg["num_layers"] >= 1
        self.h = round(math.sqrt(self.cfg["hidden_size"] // self.cfg["num_layers"]) - 1)
        core = [
            LinearAttentionBlock(
                input_size=self.cfg["preprocessor_output_size"],
                hidden_size=self.h,
                S_aggregator=self.cfg["S_aggregator"],
                Z_aggregator=self.cfg["Z_aggregator"],
                feed_forward=True,
            )
        ]
        for _ in range(self.cfg["num_layers"] - 1):
            core.append(
                LinearAttentionBlock(
                    input_size=self.h,
                    hidden_size=self.h,
                    S_aggregator=self.cfg["S_aggregator"],
                    Z_aggregator=self.cfg["Z_aggregator"],
                    feed_forward=True,
                )
            )
        self.core = nn.ModuleList(core)
        self.unmap = nn.Linear(self.h, self.cfg["hidden_size"])

    def initial_state(self) -> List[TensorType]:
        state = []
        for _ in range(self.cfg["num_layers"]):
            state += [torch.zeros(1, self.h, self.h), torch.zeros(1, self.h)]
        return state

    def forward_memory(
        self,
        z: TensorType,
        state: List[TensorType],
        t_starts: TensorType,
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        B, T, _ = z.shape
        for i, cell in enumerate(self.core):
            z, (s0, s1) = cell(z, [state[2 * i], state[2 * i + 1]])
            state[2 * i] = s0
            state[2 * i + 1] = s1
        z = self.unmap(z)
        return z, [s[:, -1:] for s in state]
