from typing import List, Tuple

import gymnasium as gym
import torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn

from popgym.baselines.ray_models.base_model import BaseModel


class GRU(BaseModel):
    r"""The gated recurrent unit from

    .. code-block:: text

        @article{chung_empirical_2014,
            title = {
                Empirical evaluation of gated recurrent neural
                networks on sequence modeling
            },
            journal = {arXiv preprint arXiv:1412.3555},
            author = {
                Chung, Junyoung and Gulcehre, Caglar and Cho,
                KyungHyun and Bengio, Yoshua
            },
            year = {2014},
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
            self.core = nn.GRUCell(
                self.cfg["preprocessor_output_size"],
                self.cfg["hidden_size"],
            )
        else:
            self.core = nn.GRU(
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

        memory = state[0].permute(1, 0, 2)

        if self.cfg["benchmark"]:
            outs = []
            memory = memory.squeeze(0)
            for t in range(z.shape[1]):
                memory = self.core(z[:, t], memory)
                outs.append(memory)
            z = torch.stack(outs, dim=1)
            memory = memory.unsqueeze(0)
        else:
            z, memory = self.core(z, memory)
        # State expected to be list
        state = [memory.permute(1, 0, 2)]  # type: ignore

        return z, state
