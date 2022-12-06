import unittest

import numpy as np

from popgym.envs.multiarmed_bandit import MultiarmedBandit
from tests.base_env_test import AbstractTest


class TestMultiarmedBandit(AbstractTest.POPGymTest):
    def setUp(self) -> None:
        self.env = MultiarmedBandit()

    def test_best(self):
        obs, info = self.env.reset(return_info=True)
        bandits = info["bandits"]
        action = np.argmax(bandits)
        done = False
        reward = 0
        expected_value = (
                                 self.env.max_episode_length * np.max(bandits)
                                 - self.env.max_episode_length * (1 - np.max(bandits))
        ) / self.env.max_episode_length

        for i in range(self.env.max_episode_length):
            self.assertFalse(done)
            obs, rew, done, info = self.env.step(action)
            reward += rew

        self.assertTrue(done)
        self.assertTrue(np.abs(expected_value - reward) < 0.1)
