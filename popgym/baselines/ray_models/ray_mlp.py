from typing import Any, Dict, List, Tuple

import gym
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn

from popgym.baselines.ray_models.base_model import BaseModel


class MLP(BaseModel):
    """A model for traditional RNNs. This supports LSTMs, Elman RNNs, and GRUs"""

    MODEL_CONFIG: Dict[str, Any] = {
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
        self.core = nn.Sequential(
            nn.Linear(self.cfg["preprocessor_output_size"], self.cfg["hidden_size"]),
            nn.LeakyReLU(),
            nn.Linear(self.cfg["hidden_size"], self.cfg["hidden_size"]),
            nn.LeakyReLU(),
        )

    def initial_state(self) -> List[TensorType]:
        return []

    def forward_memory(
        self,
        z: TensorType,
        state: List[TensorType],
        t_starts: TensorType,
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:

        z = self.core(z)

        # State expected to be list
        return z, []


class BasicMLP(MLP):
    MODEL_CONFIG = {
        "embedding": None,
    }
