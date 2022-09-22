from typing import Optional, Tuple, Union

import gym
import numpy as np

from popgym.envs.stateless_pendulum import StatelessPendulum


class NoisyStatelessPendulum(StatelessPendulum):
    def __init__(self, *args, **kwargs):
        noise_sigma = kwargs.pop("noise_sigma", 0.1)
        super().__init__(*args, **kwargs)
        self.noise_sigma = np.full_like(self.observation_space.shape, noise_sigma)

    def step(
        self, action: gym.core.ActType
    ) -> Tuple[gym.core.ObsType, float, bool, dict]:
        next_obs, reward, done, info = super().step(action)
        # TODO: Use rng from reset
        noise = np.random.normal(0, self.noise_sigma).astype(np.float32)
        next_obs = np.clip(
            next_obs + noise, self.observation_space.low, self.observation_space.high
        )
        return next_obs, reward, done, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ) -> Union[gym.core.ObsType, Tuple[gym.core.ObsType, dict]]:

        noise = np.random.normal(0, self.noise_sigma).astype(np.float32)
        if return_info:
            init_obs, info = super().reset(
                seed=seed, return_info=return_info, options=options
            )
            init_obs = np.clip(
                init_obs + noise,
                self.observation_space.low,
                self.observation_space.high,
            )
            return init_obs, info
        else:
            init_obs = super().reset(
                seed=seed, return_info=return_info, options=options
            )
            init_obs = np.clip(
                init_obs + noise,
                self.observation_space.low,
                self.observation_space.high,
            )
            # init_obs is [x-pos, x-veloc, angle, angle-veloc]
            return init_obs


class NoisyStatelessPendulumEasy(NoisyStatelessPendulum):
    def __init__(self):
        super().__init__(noise_sigma=0.1)


class NoisyStatelessPendulumMedium(NoisyStatelessPendulum):
    def __init__(self):
        super().__init__(noise_sigma=0.3)


class NoisyStatelessPendulumHard(NoisyStatelessPendulum):
    def __init__(self):
        super().__init__(noise_sigma=0.5)
