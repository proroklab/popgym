"""Partially observable variant of the CartPole gym environment.

We delete the position and angular position components of the state, so that it
can only be solved by a memory enhanced model (policy)."""

# Inspired by ray rllib at
# https://github.com/ray-project/ray/blob/master/rllib/examples/env/stateless_cartpole.py

import math
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium.spaces import Box

from popgym.core.env import POPGymEnv


class VelocityOnlyCartPole(CartPoleEnv, POPGymEnv):
    """Partially observable variant of the CartPole gym environment.

    We delete the position and angular position components of the state, so that it
    can only be solved by a memory enhanced model (policy).

    Args:
        max_episode_length: Exactly what it sounds like
    Returns:
        A gym environment
    """

    def __init__(self, *args, **kwargs):
        self.max_episode_length = kwargs.pop("max_episode_length", 200)
        super().__init__(*args, **kwargs)

        # Fix our observation-space (remove 2 velocity components).
        high = np.array(
            [
                np.inf,
                np.inf,
            ],
            dtype=np.float32,
        )
        self.state_space = self.observation_space
        self.observation_space = Box(low=-high, high=high, dtype=np.float32)

    def get_state(self):
        state = np.array(self.state, dtype=np.float32)
        return state

    def reward_transform(self, reward):
        if math.isclose(reward, 0):
            transformed = -1.0
        else:
            transformed = 1.0 / self.max_episode_length
        return transformed

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        next_obs, reward, terminated, truncated, info = super().step(action)
        self.num_steps += 1
        if self.num_steps >= self.max_episode_length:
            # Agent beat episode
            truncated = True
        reward = self.reward_transform(reward)
        # next_obs is [x-pos, x-veloc, angle, angle-veloc]
        next_obs = np.array([next_obs[1], next_obs[3]])
        return next_obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[gym.core.ObsType, dict]:
        self.num_steps = 0
        init_obs, info = super().reset(seed=seed, options=options)
        init_obs = np.array([init_obs[0], init_obs[2]])
        return init_obs, info


class VelocityOnlyCartPoleEasy(VelocityOnlyCartPole):
    pass


class VelocityOnlyCartPoleMedium(VelocityOnlyCartPole):
    def __init__(self, *args, **kwargs):
        super().__init__(max_episode_length=400)


class VelocityOnlyCartPoleHard(VelocityOnlyCartPole):
    def __init__(self, *args, **kwargs):
        super().__init__(max_episode_length=600)
