from typing import List, Tuple

import gymnasium as gym
import torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn

from popgym.baselines.ray_models.base_model import BaseModel


class Elman(BaseModel):
    """The Elman Network (RNN) from

    .. code-block:: text

        @article{elman_finding_1990,
            title = {Finding structure in time},
            volume = {14},
            number = {2},
            journal = {Cognitive science},
            author = {Elman, Jeffrey L},
            year = {1990},
            note = {Publisher: Wiley Online Library},
            pages = {179--211},
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
        if self.cfg["benchmark"]:
            self.core = nn.RNNCell(
                self.cfg["preprocessor_output_size"],
                self.cfg["hidden_size"],
            )
        else:
            self.core = nn.RNN(
                self.cfg["preprocessor_output_size"],
                self.cfg["hidden_size"],
                self.cfg["num_recurrent_layers"],
                batch_first=True,
            )

    def initial_state(self) -> List[TensorType]:
        return [torch.zeros(1, self.cfg["hidden_size"])]

    def forward_memory(
        self,
        z: TensorType,
        state: List[TensorType],
        t_starts: TensorType,
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:

        state = state[0].permute(1, 0, 2)

        if self.cfg["benchmark"]:
            outs = []
            state = state.squeeze(0)  # type: ignore
            for t in range(z.shape[1]):
                state = self.core(z[:, t], state)
                outs.append(state)
            z = torch.stack(outs, dim=1)
            state = state.unsqueeze(0)  # type: ignore
        else:
            z, state = self.core(z, state)

        state = [state.permute(1, 0, 2)]  # type: ignore

        return z, state
