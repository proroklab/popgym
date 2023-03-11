from typing import List, Tuple

import gymnasium as gym
import torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn

from popgym.baselines.ray_models.base_model import BaseModel


class Frameconv(BaseModel):
    r"""Temporal convolution over a series of observations. stack_size determines
    the temporal extent of the filter. See

    .. code-block:: text

        @article{bai_empirical_2018,
            title = {
                An empirical evaluation of generic convolutional and recurrent networks
                for sequence modeling
            },
            journal = {arXiv preprint arXiv:1803.01271},
            author = {Bai, Shaojie and Kolter, J Zico and Koltun, Vladlen},
            year = {2018},
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
        self.core = nn.Conv1d(
            self.cfg["preprocessor_output_size"],
            self.cfg["hidden_size"],
            self.cfg["stack_size"],
            1,
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

        out = self.core(time_padded.permute(0, 2, 1)).permute(0, 2, 1)
        state = time_padded[:, -self.cfg["stack_size"] + 1 :]

        # State expected to be list
        return out, [state]
