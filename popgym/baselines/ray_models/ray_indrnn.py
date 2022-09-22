from typing import List, Tuple

import gym
import torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from popgym.baselines.models.indrnn import IndRNN as IndRNNModel
from popgym.baselines.ray_models.base_model import BaseModel


class IndRNN(BaseModel):
    MODEL_CONFIG = {
        "activation": "tanh",
        "num_layers": 2,
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
        self.core = IndRNNModel(
            input_size=self.cfg["preprocessor_output_size"],
            hidden_size=self.cfg["hidden_size"],
            max_len=model_config["max_seq_len"],
        )

    def initial_state(self) -> List[TensorType]:
        return [
            torch.zeros(2, self.cfg["hidden_size"]),
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
        return z, [memory]
