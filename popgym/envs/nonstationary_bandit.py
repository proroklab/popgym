from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np

from popgym.envs.multiarmed_bandit import MultiarmedBandit


class NonstationaryBandit(MultiarmedBandit):
    def __init__(self, *args, **kwargs):
        super(NonstationaryBandit, self).__init__(*args, *kwargs)
        self.state_space = gym.spaces.Tuple(self.state_space, self.state_space)
        self.final_bandits = np.zeros((self.num_bandits,))

    def get_state(self):
        state = super(NonstationaryBandit, self).get_state()
        return state, self.final_bandits.copy()

    def step(self, action, increment=True):
        obs, reward, done, info = super().step(action, increment)
        self.bandits = (self.final_bandits - self.bandits) / self.max_episode_length
        return obs, reward, done, self.info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[gym.core.ObsType, Tuple[gym.core.ObsType, Dict[str, Any]]]:

        out = super().reset(seed=seed, return_info=return_info, options=options)
        self.final_bandits = self.np_random.rand(self.num_bandits)
        if return_info:
            obs, info = out
        else:
            obs = out
            info = {}

        info["final_bandits"] = self.final_bandits

        if return_info:
            return obs, info
        return obs


class NonstationaryBanditEasy(NonstationaryBandit):
    def __init__(self):
        super().__init__(num_bandits=10, episode_length=200)


class NonstationaryBanditMedium(NonstationaryBandit):
    def __init__(self):
        super().__init__(num_bandits=20, episode_length=400)


class NonstationaryBanditHard(NonstationaryBandit):
    def __init__(self):
        super().__init__(num_bandits=30, episode_length=600)
