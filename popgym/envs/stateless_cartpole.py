# Inspired by ray rllib at
# https://github.com/ray-project/ray/blob/master/rllib/examples/env/stateless_cartpole.py

import math
from typing import Optional, Tuple, Union

import gym
import numpy as np
from gym.envs.classic_control import CartPoleEnv
from gym.spaces import Box


class StatelessCartPole(CartPoleEnv):
    """Partially observable variant of the CartPole gym environment.
    https://github.com/openai/gym/blob/master/gym/envs/classic_control/
    cartpole.py
    We delete the x- and angular velocity components of the state, so that it
    can only be solved by a memory enhanced model (policy).
    """

    def __init__(self, *args, **kwargs):
        self.max_episode_length = kwargs.pop("max_episode_length", 200)
        super().__init__(*args, **kwargs)

        # Fix our observation-space (remove 2 velocity components).
        high = np.array(
            [
                self.x_threshold * 2,
                self.theta_threshold_radians * 2,
            ],
            dtype=np.float32,
        )

        self.observation_space = Box(low=-high, high=high, dtype=np.float32)

    def reward_transform(self, reward):
        if math.isclose(reward, 0):
            transformed = -1.0
        else:
            transformed = 1.0 / self.max_episode_length
        return transformed

    def step(
        self, action: gym.core.ActType
    ) -> Tuple[gym.core.ObsType, float, bool, dict]:
        next_obs, reward, done, info = super().step(action)
        self.num_steps += 1
        if self.num_steps >= self.max_episode_length:
            # Agent beat episode
            done = True
        reward = self.reward_transform(reward)
        # next_obs is [x-pos, x-veloc, angle, angle-veloc]
        return np.array([next_obs[0], next_obs[2]]), reward, done, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ) -> Union[gym.core.ObsType, Tuple[gym.core.ObsType, dict]]:
        self.num_steps = 0
        if return_info:
            init_obs, info = super().reset(
                seed=seed, return_info=return_info, options=options
            )
            return np.array([init_obs[0], init_obs[2]]), info
        else:
            init_obs = super().reset(
                seed=seed, return_info=return_info, options=options
            )
            # init_obs is [x-pos, x-veloc, angle, angle-veloc]
            return np.array([init_obs[0], init_obs[2]])


class StatelessCartPoleEasy(StatelessCartPole):
    pass


class StatelessCartPoleMedium(StatelessCartPole):
    def __init__(self, *args, **kwargs):
        super().__init__(max_episode_length=400)


class StatelessCartPoleHard(StatelessCartPole):
    def __init__(self, *args, **kwargs):
        super().__init__(max_episode_length=600)
