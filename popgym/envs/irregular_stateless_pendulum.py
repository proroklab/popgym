# Inspired by ray rllib at
# https://github.com/ray-project/ray/blob/master/rllib/examples/env/stateless_pendulum.py

import random
from typing import Optional, Tuple, Union

import gym
import numpy as np
from gym.spaces import Box

from popgym.envs.stateless_pendulum import StatelessPendulum


class IrregularStatelessPendulum(StatelessPendulum):
    """Partially observable variant of the Pendulum gym environment.
    https://github.com/openai/gym/blob/master/gym/envs/classic_control/
    pendulum.py
    We delete the angular velocity component of the state, so that it
    can only be solved by a memory enhanced model (policy).
    """

    def __init__(self, *args, **kwargs):
        original_dt = 0.02
        self.dt_range = kwargs.pop("tau_range", (0.02, 0.03))
        max_diff = max(self.dt_range) / original_dt
        super().__init__(*args, **kwargs)

        # Fix our observation-space (remove angular velocity component).
        high = np.array([1.0, 1.0, max(self.dt_range)], dtype=np.float32)
        self.observation_space = Box(
            low=-high * max_diff, high=high * max_diff, dtype=np.float32
        )

    def step(
        self, action: gym.core.ActType
    ) -> Tuple[gym.core.ObsType, float, bool, dict]:
        self.dt = random.uniform(*self.dt_range)
        next_obs, reward, done, info = super().step(action)
        return np.array([*next_obs, self.dt], dtype=np.float32), reward, done, info

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
            return np.array([*init_obs, self.dt], dtype=np.float32), info

        else:
            init_obs = super().reset(
                seed=seed, return_info=return_info, options=options
            )
            # init_obs is [cos(theta), sin(theta), theta-dot (angular velocity)]
            return np.array([*init_obs, self.dt], dtype=np.float32)


class IrregularStatelessPendulumEasy(IrregularStatelessPendulum):
    pass


class IrregularStatelessPendulumMedium(IrregularStatelessPendulum):
    def __init__(self, *args, **kwargs):
        super().__init__(tau_range=(0.02, 0.04))


class IrregularStatelessPendulumHard(IrregularStatelessPendulum):
    def __init__(self, *args, **kwargs):
        super().__init__(tau_range=(0.02, 0.05))
