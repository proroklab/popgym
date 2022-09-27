from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np


class MultiarmedBandit(gym.Env):
    """Multiarmed Bandits over an episode. Bandits are initialized that have some
    probability of positive reward and negative reward. The agent must sample
    and then exploit the bandits that pay best.

    Args:
        num_bandits: Number of individual bandits
        episode_length: Max episode length
    Returns:
        A gym env
    """

    def __init__(self, num_bandits=10, episode_length=200):
        self.num_bandits = num_bandits
        self.episode_length = episode_length
        self.observation_space = gym.spaces.Tuple(
            (gym.spaces.MultiDiscrete(np.array([num_bandits, 2])))
        )
        self.action_space = gym.spaces.Discrete(num_bandits)

    def step(self, action, increment=True):
        value = int(np.random.rand() < self.bandits[action])
        obs = np.array([action, value], dtype=np.int32)
        if value:
            reward = 1 / self.episode_length
        else:
            reward = -1 / self.episode_length
        done = self.num_steps >= self.episode_length - 1
        if increment:
            self.num_steps += 1

        return obs, reward, done, self.info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[gym.core.ObsType, Tuple[gym.core.ObsType, Dict[str, Any]]]:

        if seed is not None:
            np.random.seed(seed)

        self.num_steps = 0
        self.bandits = np.random.rand(self.num_bandits)
        self.info = {"bandits": self.bandits}
        rand_action = np.random.randint(self.num_bandits)
        obs, _, _, info = self.step(rand_action, increment=False)

        if return_info:
            return obs, info
        return obs


class MultiarmedBanditEasy(MultiarmedBandit):
    def __init__(self):
        super().__init__(num_bandits=10, episode_length=200)


class MultiarmedBanditMedium(MultiarmedBandit):
    def __init__(self):
        super().__init__(num_bandits=20, episode_length=400)


class MultiarmedBanditHard(MultiarmedBandit):
    def __init__(self):
        super().__init__(num_bandits=30, episode_length=600)
