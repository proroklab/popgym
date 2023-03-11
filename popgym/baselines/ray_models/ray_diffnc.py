from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import gymnasium as gym
import torch
from dnc import DNC
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from popgym.baselines.ray_models.base_model import BaseModel


class DiffNC(BaseModel):
    r"""The differentiable neural computer from

    .. code-block:: text

        @techreport{wayne_unsupervised_2018,
            title = {Unsupervised {Predictive} {Memory} in a {Goal}-{Directed} {Agent}},
            url = {http://arxiv.org/abs/1803.10760},
            number = {arXiv:1803.10760},
            urldate = {2022-09-09},
            institution = {arXiv},
            month = mar,
            year = {2018},
            doi = {10.48550/arXiv.1803.10760},
        }
    """

    MODEL_CONFIG = {
        # Number of controller hidden layers
        "num_hidden_layers": 1,
        # Number of LSTM units
        "num_layers": 1,
        # Number of read heads, i.e. how many addrs are read at once
        "read_heads": 4,
        # Size of each cell, also
        # number of cells == hidden_size // cell_size
        "cell_size": 16,
        # LSTM activation function
        "nonlinearity": "tanh",
    }

    MEMORY_KEYS = [
        "memory",
        "link_matrix",
        "precedence",
        "read_weights",
        "write_weights",
        "usage_vector",
    ]

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
        self.cfg["nr_cells"] = self.cfg["hidden_size"] // self.cfg["cell_size"]
        self.dnc_built = False

    def initial_state(self) -> List[TensorType]:
        ctrl_hidden = [
            torch.zeros(self.cfg["num_hidden_layers"], self.cfg["hidden_size"]),
            torch.zeros(self.cfg["num_hidden_layers"], self.cfg["hidden_size"]),
        ]
        m = self.cfg["nr_cells"]
        r = self.cfg["read_heads"]
        w = self.cfg["cell_size"]
        memory = [
            torch.zeros(m, w),  # memory
            torch.zeros(1, m, m),  # link_matrix
            torch.zeros(1, m),  # precedence
            torch.zeros(r, m),  # read_weights
            torch.zeros(1, m),  # write_weights
            torch.zeros(m),  # usage_vector
        ]

        read_vecs = torch.zeros(w * r)

        state = [*ctrl_hidden, read_vecs, *memory]
        return state

    def unpack_state(
        self,
        state: List[TensorType],
    ) -> Tuple[List[Tuple[TensorType, TensorType]], Dict[str, TensorType], TensorType]:
        """Given a list of tensors, reformat for self.dnc input"""
        assert len(state) == 9, "Failed to verify unpacked state"
        ctrl_hidden: List[Tuple[TensorType, TensorType]] = [
            (
                state[0].permute(1, 0, 2).contiguous(),
                state[1].permute(1, 0, 2).contiguous(),
            )
        ]
        read_vecs: TensorType = state[2]
        memory: List[TensorType] = state[3:]
        memory_dict: OrderedDict[str, TensorType] = OrderedDict(
            zip(self.MEMORY_KEYS, memory)
        )

        return ctrl_hidden, memory_dict, read_vecs

    def pack_state(
        self,
        ctrl_hidden: List[Tuple[TensorType, TensorType]],
        memory_dict: Dict[str, TensorType],
        read_vecs: TensorType,
    ) -> List[TensorType]:
        """Given the dnc output, pack it into a list of tensors
        for rllib state. Order is ctrl_hidden, read_vecs, memory_dict"""
        state = []
        ctrl_hidden = [
            ctrl_hidden[0][0].permute(1, 0, 2),
            ctrl_hidden[0][1].permute(1, 0, 2),
        ]
        state += ctrl_hidden
        assert len(state) == 2, "Failed to verify packed state"
        state.append(read_vecs)
        assert len(state) == 3, "Failed to verify packed state"
        state += memory_dict.values()
        assert len(state) == 9, "Failed to verify packed state"
        return state

    def build_dnc(self, device_idx: Union[int, None]) -> torch.nn.Module:
        return DNC(
            input_size=self.cfg["preprocessor_input_size"],
            hidden_size=self.cfg["hidden_size"],
            num_layers=self.cfg["num_layers"],
            num_hidden_layers=self.cfg["num_hidden_layers"],
            read_heads=self.cfg["read_heads"],
            cell_size=self.cfg["cell_size"],
            nr_cells=self.cfg["nr_cells"],
            nonlinearity=self.cfg["nonlinearity"],
            gpu_id=device_idx,
        )

    def forward_memory(
        self,
        z: TensorType,
        state: List[TensorType],
        t_starts: TensorType,
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        B, T, F = z.shape

        # First run
        if not self.dnc_built:
            gpu_id = z.device.index if z.device.index is not None else -1
            self.core = self.build_dnc(gpu_id)
            self.core.output = torch.nn.Linear(
                self.core.output.in_features, self.cfg["hidden_size"]
            ).to(z.device)
            self.dnc_built = True

        unpacked = self.unpack_state(state)
        output, unpacked = self.core(z, unpacked)
        packed = self.pack_state(*unpacked)

        return output, packed
