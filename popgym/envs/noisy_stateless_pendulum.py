from typing import Optional, Tuple, Union

import gym
import numpy as np

from popgym.envs.stateless_pendulum import StatelessPendulum


class NoisyStatelessPendulum(StatelessPendulum):
    def __init__(self, *args, **kwargs):
        noise_sigma = kwargs.pop("noise_sigma", 0.1)
        super().__init__(*args, **kwargs)
        self.noise_sigma = np.full(self.observation_space.shape, noise_sigma)

    def add_noise_to_obs(self, obs):
        noise = self.np_random.normal(0, self.noise_sigma).astype(np.float32)
        obs_space = self.observation_space
        obs = np.clip(
            obs + noise, obs_space.low, obs_space.high
        )
        return obs

    def step(
        self, action: gym.core.ActType
    ) -> Tuple[gym.core.ObsType, float, bool, dict]:
        next_obs, reward, done, info = super().step(action)
        return self.add_noise_to_obs(next_obs), reward, done, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ) -> Union[gym.core.ObsType, Tuple[gym.core.ObsType, dict]]:

        if return_info:
            init_obs, info = super().reset(
                seed=seed, return_info=return_info, options=options
            )
            init_obs = self.add_noise_to_obs(init_obs)
            return init_obs, info
        else:
            init_obs = super().reset(
                seed=seed, return_info=return_info, options=options
            )
            init_obs = self.add_noise_to_obs(init_obs)
            return init_obs


class NoisyStatelessPendulumEasy(NoisyStatelessPendulum):
    def __init__(self):
        super().__init__(noise_sigma=0.1)


class NoisyStatelessPendulumMedium(NoisyStatelessPendulum):
    def __init__(self):
        super().__init__(noise_sigma=0.2)


class NoisyStatelessPendulumHard(NoisyStatelessPendulum):
    def __init__(self):
        super().__init__(noise_sigma=0.3)
