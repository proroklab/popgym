from typing import List, Tuple

import gymnasium as gym
import torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from popgym.baselines.models.lmu import LMU as LMUModel
from popgym.baselines.ray_models.base_model import BaseModel


class LMU(BaseModel):
    r"""Legendre Memory Units from

    .. code-block:: text

        @inproceedings{voelker_legendre_2019,
            title = {
                Legendre Memory Units: Continuous-Time
                Representation in Recurrent Neural Networks
            },
            volume = {32},
            shorttitle = {Legendre {Memory} {Units}},
            urldate = {2022-09-22},
            booktitle = {Advances in {Neural} {Information} {Processing} {Systems}},
            publisher = {Curran Associates, Inc.},
            author = {Voelker, Aaron and Kajić, Ivana and Eliasmith, Chris},
            year = {2019},
        }
    """

    MODEL_CONFIG = {
        "embedding": None,
        "theta": 64,
        "learn_a": False,
        "learn_b": False,
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
        self.core = LMUModel(
            input_size=self.cfg["preprocessor_output_size"],
            hidden_size=self.cfg["hidden_size"],
            memory_size=self.cfg["hidden_size"],
            theta=self.cfg["theta"],
            learn_a=self.cfg["learn_a"],
            learn_b=self.cfg["learn_b"],
        )

    def initial_state(self) -> List[TensorType]:
        return [
            torch.zeros(1, self.cfg["hidden_size"]),
            torch.zeros(1, self.cfg["hidden_size"]),
        ]

    def forward_memory(
        self,
        z: TensorType,
        state: List[TensorType],
        t_starts: TensorType,
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:

        hidden, memory = state

        z, (hidden, memory) = self.core(z, (hidden.squeeze(1), memory.squeeze(1)))

        # State expected to be list
        return z, [hidden.unsqueeze(1), memory.unsqueeze(1)]
