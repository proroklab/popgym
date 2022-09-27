import math
from typing import List, Tuple

import gym
import torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn

from popgym.baselines.models.fwp import FWPBlock
from popgym.baselines.ray_models.base_model import BaseModel


class FastWeightProgrammer(BaseModel):
    r"""The fast weight programmer from

    @inproceedings{schlag_linear_2021,
        title = {Linear {Transformers} {Are} {Secretly} {Fast} {Weight} {Programmers}},
        url = {https://proceedings.mlr.press/v139/schlag21a.html},
        language = {en},
        urldate = {2022-09-21},
        booktitle = {
            Proceedings of the 38th International Conference on Machine Learning
        },
        publisher = {PMLR},
        author = {Schlag, Imanol and Irie, Kazuki and Schmidhuber, Jürgen},
        month = jul,
        year = {2021},
        note = {ISSN: 2640-3498},
        pages = {9355--9366},
    }

    without the RNN extensions."""

    MODEL_CONFIG = {"sum_normalization": True, "embedding": "sine", "aggregator": "sum"}

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
        self.h = round(math.sqrt(self.cfg["hidden_size"]))
        self.core = FWPBlock(
            input_size=self.cfg["preprocessor_output_size"],
            hidden_size=self.h,
            aggregator=self.cfg["aggregator"],
            sum_normalization=self.cfg["sum_normalization"],
        )
        self.unmap = nn.Linear(self.h, self.cfg["hidden_size"])

    def initial_state(self) -> List[TensorType]:
        return [
            torch.zeros(1, self.h, self.h),
        ]

    def forward_memory(
        self,
        z: TensorType,
        state: List[TensorType],
        t_starts: TensorType,
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        B, T, _ = z.shape
        [memory] = state
        z, memory = self.core(z, memory)
        h = z.shape[-1]
        y = self.unmap(z)
        return y, [
            memory[:, -1].reshape(B, 1, h, h),
        ]
