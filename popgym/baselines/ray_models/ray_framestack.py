from typing import List, Tuple

import gymnasium as gym
import torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn

from popgym.baselines.ray_models.base_model import BaseModel


class Framestack(BaseModel):
    r"""Concatenate sequential frames along the feature dimension,
    pushing them through an MLP. From

    .. code-block:: text

        @article{mnih_human-level_2015,
            title = {Human-level control through deep reinforcement learning},
            volume = {518},
            number = {7540},
            journal = {nature},
            author = {
                Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Rusu,
                Andrei A and Veness, Joel and Bellemare, Marc G and Graves,
                Alex and Riedmiller, Martin and Fidjeland, Andreas K and Ostrovski,
                Georg and {others}
            },
            year = {2015},
            note = {Publisher: Nature Publishing Group},
            pages = {529--533},
        }
    """

    MODEL_CONFIG = {
        # Number of consecutive frames to stack into a single input
        "stack_size": 4,
        "embedding": "sine",
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
        # Compress before passing to framestack so we are not "cheating"
        # by using a larger hidden state than other models
        self.compress = nn.Linear(
            self.cfg["preprocessor_output_size"],
            self.cfg["hidden_size"] // self.cfg["stack_size"],
        )
        self.core = nn.Sequential(
            nn.Linear(
                self.cfg["hidden_size"],
                self.cfg["hidden_size"],
            ),
            nn.LeakyReLU(),
            nn.Linear(self.cfg["hidden_size"], self.cfg["hidden_size"]),
            nn.LeakyReLU(),
        )

    def initial_state(self) -> List[TensorType]:
        return [
            torch.zeros(
                self.cfg["stack_size"] - 1, self.cfg["preprocessor_output_size"]
            ),
        ]

    def forward_memory(
        self,
        z: TensorType,
        state: List[TensorType],
        t_starts: TensorType,
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:

        assert self.cfg["hidden_size"] % self.cfg["stack_size"] == 0, (
            f"Hidden size {self.cfg['hidden_size']} must be divisible by stack size"
            f" {self.cfg['stack_size']}"
        )
        B, T, F = z.shape
        # [B, T + stack_size - 1, F]
        time_padded = torch.cat([state[0], z], dim=1)
        # [B, T + stack_size - 1, f, k * T]
        unfolded = time_padded.unfold(1, self.cfg["stack_size"], 1)
        # [B * T, stack_size, k]
        stacked = unfolded.permute(0, 1, 3, 2).reshape(
            B * T, self.cfg["stack_size"], z.shape[-1]
        )
        # Add extra dim so compression is applied frame-wise, then fold
        # new dim back into feature dim
        compressed = self.compress(
            stacked.reshape(
                B, T, self.cfg["stack_size"], self.cfg["preprocessor_output_size"]
            )
        ).reshape(B, T, self.cfg["hidden_size"])
        z = self.core(compressed)
        # Output last stack_size - 1 from stacked at time T
        state = stacked.reshape(
            B, T, self.cfg["stack_size"], self.cfg["preprocessor_output_size"]
        )[:, -1, 1:]

        # State expected to be list
        return z, [state]
