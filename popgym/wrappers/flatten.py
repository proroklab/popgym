from typing import Tuple

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType

from popgym.core.env import POPGymEnv
from popgym.core.wrapper import POPGymWrapper


class Flatten(POPGymWrapper):
    """
    Wrapper that flattens the observation and action spaces
    to make them compatible with neural networks.
    """

    def __init__(
        self,
        env: POPGymEnv,
        flatten_action: bool = True,
        flatten_observation: bool = True,
    ):
        super().__init__(env)
        self.need_flatten_action = False
        self.need_flatten_obs = False
        if flatten_action:
            # Check that spaces are either all continuous or all discrete
            self.continuous(self.env.action_space)
            self.need_flatten_action = isinstance(
                self.env.action_space, gym.spaces.Tuple
            ) or isinstance(self.env.action_space, gym.spaces.Dict)
            if self.need_flatten_action:
                self.action_space = gym.spaces.utils.flatten_space(
                    self.env.action_space
                )
        if flatten_observation:
            self.need_flatten_obs = isinstance(
                self.env.observation_space, gym.spaces.Tuple
            ) or isinstance(self.env.observation_space, gym.spaces.Dict)
            if self.need_flatten_obs:
                self.observation_space = gym.spaces.utils.flatten_space(
                    self.env.observation_space
                )

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        if self.need_flatten_action:
            flat_action = gym.spaces.utils.unflatten(self.env.action_space, action)
        else:
            flat_action = action

        obs, reward, terminated, truncated, info = self.env.step(flat_action)

        if self.need_flatten_obs:
            obs = gym.spaces.utils.flatten(self.env.observation_space, obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.need_flatten_obs:
            obs = gym.spaces.utils.flatten(self.env.observation_space, obs)
        return obs, info

    def continuous(self, space: gym.spaces.Space) -> bool:
        if isinstance(space, gym.spaces.Box):
            if np.issubdtype(space.dtype, np.integer):
                return False
            else:
                return True
        elif isinstance(
            space,
            (gym.spaces.Discrete, gym.spaces.MultiDiscrete, gym.spaces.MultiBinary),
        ):
            return False
        elif isinstance(space, gym.spaces.Tuple):
            res = [self.continuous(s) for s in space.spaces]
            assert all(res) or not any(res), "Cannot mix continuous and discrete spaces"
            return any(res)
        elif isinstance(space, gym.spaces.Dict):
            res = [self.continuous(s) for s in space.spaces.values()]
            assert all(res) or not any(res), "Cannot mix continuous and discrete spaces"
            return any(res)
        else:
            raise NotImplementedError(f"Unknown space: {space}")
