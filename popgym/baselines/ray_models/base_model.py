from typing import Any, Dict, List, Tuple

import gymnasium as gym
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn

from popgym.baselines.models.embeddings import (
    Base2Embedding,
    PerceptronEmbedding,
    ScaledPositionalEncoding,
    SimpleScaledEncoding,
    Time2Vec,
    positional_encoding,
)


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight)


class BaseModel(TorchModelV2, nn.Module):
    """The base memory model that all other memory models should
    inherit from.

    You will need to override initial_state and
    memory forward, as well as set your model defaults using
    MODEL_CONFIG.
    """

    BASE_CONFIG = {
        # Number of weights per controller hidden layer
        "hidden_size": 256,
        # Observation goes through this torch.nn.Module before
        # feeding to memory
        "preprocessor": torch.nn.Identity(),
        "postprocessor": torch.nn.Identity(),
        "actor": nn.Sequential(nn.Linear(256, 256), nn.LeakyReLU(inplace=True)),
        "critic": nn.Sequential(nn.Linear(256, 256), nn.LeakyReLU(inplace=True)),
        # The output size of the preprocessor
        "preprocessor_input_size": 256,
        # Input size to the preprocessor
        "preprocessor_output_size": 256,
        "postprocessor_output_size": 256,
        # Either none, learned, absolute, or relative
        "embedding": None,
        # Dims to use for embedding
        # if set to None, uses the entire feature vector
        "embedding_size": None,
        # Either add or concatenate the embedding
        "embedding_mode": "add",
        # Whether to learn the embedding scale
        "scale_embedding": True,
    }
    # Override me in subclasses
    MODEL_CONFIG: Dict[str, Any] = {}

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **custom_model_kwargs,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.num_outputs = num_outputs
        self.obs_dim = gym.spaces.utils.flatdim(obs_space)
        self.act_space = action_space
        self.act_dim = gym.spaces.utils.flatdim(action_space)

        self.input_dim = self.obs_dim
        for k in custom_model_kwargs:
            assert (
                k in self.BASE_CONFIG or k in self.MODEL_CONFIG
            ), f"Invalid config arg {k}"
        self.cfg = {
            **self.BASE_CONFIG,
            **self.MODEL_CONFIG,
            **custom_model_kwargs,
            **model_config.get("custom_model_config", {}),
        }
        self.feature_map = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.cfg["preprocessor_input_size"]),
            torch.nn.LayerNorm(
                self.cfg["preprocessor_input_size"], elementwise_affine=False
            ),
        )
        # When rollout_mode != complete_episodes, we can have significantly
        # longer rollout fragments than max_seq_len due to additional zero-padding.
        # To fix, pass 2 * max_seq_len to setup_embedding to account for this
        # but this uses more memory and results in lower model accuracy...
        self.setup_embedding(model_config["max_seq_len"])
        self.max_seq_len = model_config["max_seq_len"]
        if self.cfg["scale_embedding"]:
            self.embedding_strength = nn.Parameter(torch.tensor([0.5]))
        self.pre = self.cfg["preprocessor"]
        self.core = nn.Identity()
        self.post = self.cfg["postprocessor"]

        actor_final = nn.Linear(self.cfg["postprocessor_output_size"], self.num_outputs)
        critic_final = nn.Linear(self.cfg["postprocessor_output_size"], 1)

        self.feature_map.apply(weight_init)
        self.pre.apply(weight_init)
        self.post.apply(weight_init)
        self.cfg["actor"].apply(weight_init)
        self.cfg["critic"].apply(weight_init)
        nn.init.normal_(actor_final.weight, std=0.01)
        nn.init.normal_(critic_final.weight, std=0.01)

        self.actor = nn.Sequential(self.cfg["actor"], actor_final)
        self.critic = nn.Sequential(self.cfg["critic"], critic_final)

        self.features = None

    def setup_embedding(self, max_seq_len: int):
        """Setup positional embedding module

        Args:
            max_seq_len (int): Maximum sequence length
        """
        assert self.cfg["embedding_mode"] in [
            "cat",
            "add",
        ], "Embedding mode must be 'add' or 'cat'"
        if self.cfg["embedding_mode"] == "cat":
            assert (
                self.cfg["embedding_size"] is not None
            ), "Must specify embedding size with 'cat' embedding mode"
        size = self.cfg["embedding_size"] or self.cfg["preprocessor_output_size"]
        if self.cfg["embedding"] == "learned":
            self.embedding = nn.Embedding(max_seq_len, size)
        elif self.cfg["embedding"] == "fixed":
            self.embedding = nn.Embedding(max_seq_len, size)
            self.embedding.requires_grad = False
            self.embedding.weight.requires_grad = False
        elif self.cfg["embedding"] == "sine":
            self.embedding = nn.Embedding.from_pretrained(
                positional_encoding(max_seq_len, size)
            )
        elif self.cfg["embedding"] == "scaled_sine":
            self.embedding = ScaledPositionalEncoding.from_pretrained(
                positional_encoding(max_seq_len, size)
            )
        elif self.cfg["embedding"] == "simple_scaled":
            self.embedding = SimpleScaledEncoding(max_seq_len, size)
        elif self.cfg["embedding"] == "perceptron":
            self.embedding = PerceptronEmbedding(max_seq_len, size)
        elif self.cfg["embedding"] == "base2":
            self.embedding = Base2Embedding(max_seq_len, size)
        elif self.cfg["embedding"] == "time2vec":
            self.embedding = Time2Vec(max_seq_len, size)
        elif self.cfg["embedding"] is None:
            self.embedding = None
        else:
            raise NotImplementedError()

    def initial_state(self) -> List[TensorType]:
        """Return the initial states for your memory model.

        The shape of states returned here should NOT contain
        the batch dimension. The batch dimension will be prepended
        by RLlib, depending on the number of episodes/rollouts
        in the batch

        Returns:
            List of tensors denoting the t=0 recurrent state"""

        raise NotImplementedError()

    def get_initial_state(self) -> List[TensorType]:
        return self.initial_state() + [
            torch.tensor([0], dtype=torch.long),
        ]

    def value_function(self) -> TensorType:
        assert self.features is not None, "must call forward() first"
        value = self.critic(self.features).reshape(-1)
        return value

    def preprocess_obs(self, obs, seq_lens):
        """Preprocess the incoming observation of shape [?, Feature] into
        [Batch, Time, Feature]
        """
        if self.view_requirements["obs"].shift == 0:
            # Recurrent models do not need to shift the observations
            B = seq_lens.shape[0]
            orig_T = obs.shape[0] // B
            unflat = obs.reshape(B, orig_T, *self.view_requirements["obs"].space.shape)
            # Sometimes RLlib sticks on extra padding for no reason
            # Remove that padding
            T = seq_lens.max().item()
            unflat = unflat[:, :T]

        else:
            # We are in attention-mode and will always receive
            # obs of shape [B, max_seq_len, F]
            B = seq_lens.shape[0]
            T = len(self.view_requirements["obs"].shift_arr)
            orig_T = T
            unflat = obs.reshape(B, T, *self.view_requirements["obs"].space.shape)

        return unflat, B, T, orig_T

    def forward_memory(
        self,
        z: TensorType,
        state: List[TensorType],
        t_starts: TensorType,
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """Forward for your custom memory model.

        Args:
            z: Preprocessed features of shape [B, T, F], with padding along the time
                dimension
            state: Recurrent states of shape [B, ...]
            t_starts: Tensor of size [B] denoteint the length of the rollout so far.
                Note that in some cases RLlib might chunk a long rollout into multiple
                forward passes. This tracks the length throughout all forward passes.
            seq_lens: Tensor of size [B] denoting the number of non-padding elements
                in z. E.g. seq_lens == [100, 60], z.shape[1] == 128 means the first 100
                elements of the first batch dimension are valid and the first 60
                elements of the second batch dimension are valid. Rest are padding.

        Returns:
            (output, state)
                where output is [B, T, D] and state is
                [B, ...]. Note that the padding must be present in the output.
                The state must be exactly the same shape as the input state.
        """

        raise NotImplementedError()

    def apply_embedding(self, x, t_starts):
        """Applies temporal embeddings to x

        Args:
            x: Tensor of shape [B, T, F]
            t_starts: Tensor of shape [B] denoting the time offset
                of the first element in x

        Returns:
            Tensor of shape [B, T, F] with the temporal embeddings
        """
        B, T, F = x.shape
        # indices used to include padding
        # note that T could be > seq_lens.max() due to shoddy RLlib batch-splitting
        time_idx = t_starts.reshape(-1, 1) + torch.arange(T, device=x.device).reshape(
            1, -1
        )

        emb = self.embedding(time_idx).reshape(B, T, -1)

        if self.cfg["scale_embedding"]:
            self.embedding_strength.data.clamp_(0.0, 1.0)
            x = x * (1.0 - self.embedding_strength)
            emb = emb * self.embedding_strength

        if self.cfg["embedding_mode"] == "add":
            if self.cfg["embedding_size"] is None:
                x = x + emb
            else:
                x[:, :, : self.cfg["embedding_size"]] = (
                    x[:, :, : self.cfg["embedding_size"]] + emb
                )
        else:
            x[:, :, : self.cfg["embedding_size"]] = emb

        return x

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        # Sometimes int64, sometimes int32
        seq_lens = seq_lens.long()
        # This is always float, fucking rllib doesn't respect dtype
        state[-1] = state[-1].long()
        x, B, T, orig_t = self.preprocess_obs(input_dict["obs_flat"], seq_lens)
        # Bug, RLlib will sometimes fail to set self.train() during training
        # and LSTM fails to backprop when self.train() not set
        self.train() if torch.is_grad_enabled() else self.eval()

        # Start and end t indices into the padded sequence
        # denoting non-padded elements
        t_starts = state[-1].squeeze(1)
        t_ends = t_starts + seq_lens

        x = self.feature_map(x)
        if self.cfg["embedding"] is not None:
            x = self.apply_embedding(x, t_starts)
        x = self.pre(x)
        x, state = self.forward_memory(x, state[:-1], t_starts, seq_lens)
        assert x.shape == (B, T, self.cfg["hidden_size"]), (
            f"Expected memory to output shape {(B, T, self.cfg['hidden_size'])} but got"
            f" {x.shape}"
        )

        features = self.post(x)
        # Add the extraneous zero-padding back (RLlib needs it)
        self.features = torch.cat(
            [
                features,
                torch.zeros(B, orig_t - T, features.shape[-1], device=x.device),
            ],
            dim=1,
        )
        logits = self.actor(self.features).reshape(B * orig_t, self.num_outputs)

        return logits, state + [t_ends.unsqueeze(1)]
