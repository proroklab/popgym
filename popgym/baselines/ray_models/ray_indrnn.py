from typing import List, Tuple

import gymnasium as gym
import torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from popgym.baselines.models.indrnn import IndRNN as IndRNNModel
from popgym.baselines.ray_models.base_model import BaseModel


class IndRNN(BaseModel):
    r"""A two-layer independently recurrent neural networks from

    .. code-block:: text

        @inproceedings{li_independently_2018,
            address = {Salt Lake City, UT},
            title = {
                Independently Recurrent Neural Network (IndRNN):
                Building a Longer and Deeper RNN
            },
            isbn = {978-1-5386-6420-9},
            shorttitle = {Independently {Recurrent} {Neural} {Network} ({IndRNN})},
            url = {https://ieeexplore.ieee.org/document/8578670/},
            doi = {10.1109/CVPR.2018.00572},
            language = {en},
            urldate = {2022-09-21},
            booktitle = {
                2018 {IEEE}/{CVF} {Conference} on {Computer} {Vision}
                and {Pattern} {Recognition}
            },
            publisher = {IEEE},
            author = {
                Li, Shuai and Li, Wanqing and Cook, Chris and Zhu, Ce and Gao, Yanbo
            },
            month = jun,
            year = {2018},
            pages = {5457--5466},
        }
    """

    MODEL_CONFIG = {
        "activation": "tanh",
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
