# Inspired by ray rllib at
# https://github.com/ray-project/ray/blob/master/rllib/examples/env/stateless_cartpole.py

import random
from typing import Optional, Tuple, Union

import gym
import numpy as np
from gym.spaces import Box

from popgym.envs.stateless_cartpole import StatelessCartPole


class IrregularStatelessCartPole(StatelessCartPole):
    """Partially observable variant of the CartPole gym environment.
    https://github.com/openai/gym/blob/master/gym/envs/classic_control/
    cartpole.py
    We delete the x- and angular velocity components of the state, so that it
    can only be solved by a memory enhanced model (policy).
    """

    def __init__(self, *args, **kwargs):
        original_tau = 0.02
        self.tau_range = kwargs.pop("tau_range", (0.02, 0.03))
        max_diff = max(self.tau_range) / original_tau
        super().__init__(*args, **kwargs)

        # Fix our observation-space (remove 2 velocity components).
        high = np.array(
            [
                self.x_threshold * 2,
                self.theta_threshold_radians * 2,
                max(self.tau_range),
            ],
            dtype=np.float32,
        )

        self.observation_space = Box(
            low=-high * max_diff, high=high * max_diff, dtype=np.float32
        )

    def step(
        self, action: gym.core.ActType
    ) -> Tuple[gym.core.ObsType, float, bool, dict]:
        self.tau = random.uniform(*self.tau_range)
        next_obs, reward, done, info = super().step(action)
        # next_obs is [x-pos, x-veloc, angle, angle-veloc]
        return np.array([*next_obs, self.tau], dtype=np.float32), reward, done, info

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
            return np.array([*init_obs, self.tau], dtype=np.float32), info
        else:
            init_obs = super().reset(
                seed=seed, return_info=return_info, options=options
            )
            # init_obs is [x-pos, x-veloc, angle, angle-veloc]
            return np.array([*init_obs, self.tau], dtype=np.float32)


class IrregularStatelessCartPoleEasy(IrregularStatelessCartPole):
    pass


class IrregularStatelessCartPoleMedium(IrregularStatelessCartPole):
    def __init__(self, *args, **kwargs):
        kwargs["tau_range"] = (0.02, 0.04)
        super().__init__(*args, **kwargs)


class IrregularStatelessCartPoleHard(IrregularStatelessCartPole):
    def __init__(self, *args, **kwargs):
        kwargs["tau_range"] = (0.02, 0.05)
        super().__init__(*args, **kwargs)
