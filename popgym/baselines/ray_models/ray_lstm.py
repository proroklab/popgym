from typing import List, Tuple

import gymnasium as gym
import torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn

from popgym.baselines.ray_models.base_model import BaseModel


class LSTM(BaseModel):
    r"""Long Short-Term Memory from

    .. code-block:: text

        @article{hochreiter_long_1997,
            title = {Long Short-Term Memory},
            volume = {9},
            issn = {0899-7667},
            url = {https://doi.org/10.1162/neco.1997.9.8.1735},
            doi = {10.1162/neco.1997.9.8.1735},
            number = {8},
            journal = {Neural Comput.},
            author = {Hochreiter, Sepp and Schmidhuber, Jürgen},
            month = nov,
            year = {1997},
            pages = {1735--1780},
        }
    """

    MODEL_CONFIG = {
        # Number of recurrent hidden layers in encoder/decoder
        "num_recurrent_layers": 1,
        "benchmark": False,
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
        # Need to define self.core
        if self.cfg["benchmark"]:
            self.core = nn.LSTMCell(
                self.cfg["preprocessor_output_size"],
                self.cfg["hidden_size"],
            )
        else:
            self.core = nn.LSTM(
                self.cfg["preprocessor_output_size"],
                self.cfg["hidden_size"],
                self.cfg["num_recurrent_layers"],
                batch_first=True,
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

        state = [s.permute(1, 0, 2) for s in state]

        if self.cfg["benchmark"]:
            outs = []
            state = [s.squeeze(0) for s in state]
            for t in range(z.shape[1]):
                state = self.core(z[:, t], state)
                outs.append(state[0])
            z = torch.stack(outs, dim=1)
            state = [s.unsqueeze(0) for s in state]
        else:
            z, state = self.core(z, state)

        state = [s.permute(1, 0, 2) for s in state]
        return z, state
