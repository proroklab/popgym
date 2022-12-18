from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np

from popgym.core.env import POPGymEnv


class MultiarmedBandit(POPGymEnv):
    """Multiarmed Bandits over an episode. Bandits are initialized that have some
    probability of positive reward and negative reward. The agent must sample
    and then exploit the bandits that pay best.

    Args:
        num_bandits: Number of individual bandits
        episode_length: Max episode length
    Returns:
        A gym env
    """

    obs_requires_prev_action = True

    def __init__(self, num_bandits=10, episode_length=200):
        self.num_bandits = num_bandits
        self.max_episode_length = episode_length
        self.observation_space = gym.spaces.Discrete(2)
        self.state_space = gym.spaces.Box(0, 1, (self.num_bandits,))
        self.action_space = gym.spaces.Discrete(num_bandits)
        self.num_steps = 0
        self.bandits = np.zeros((self.num_bandits,), dtype=np.float32)

    def get_state(self):
        return self.bandits.copy()

    def step(self, action, increment=True):
        obs = int(self.np_random.random() < self.bandits[action])
        if obs:
            reward = 1 / self.max_episode_length
        else:
            reward = -1 / self.max_episode_length
        done = self.num_steps >= self.max_episode_length - 1
        if increment:
            self.num_steps += 1

        return obs, reward, done, self.info.copy()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[gym.core.ObsType, Tuple[gym.core.ObsType, Dict[str, Any]]]:

        super(MultiarmedBandit, self).reset(seed=seed)
        self.num_steps = 0
        self.bandits = self.np_random.random(self.num_bandits, dtype=np.float32)
        self.info = {"bandits": self.bandits.copy()}
        # rand_action = np.random.randint(self.num_bandits)
        # obs, _, _, info = self.step(rand_action, increment=False)
        obs = 0

        if return_info:
            return obs, self.info
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
