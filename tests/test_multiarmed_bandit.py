import unittest

import numpy as np

from popgym.envs.multiarmed_bandit import MultiarmedBandit


class TestMultiarmedBandit(unittest.TestCase):
    def test_best(self):
        e = MultiarmedBandit()
        obs, info = e.reset(return_info=True)
        bandits = info["bandits"]
        action = np.argmax(bandits)
        done = False
        reward = 0
        expected_value = (
            e.episode_length * np.max(bandits)
            - e.episode_length * (1 - np.max(bandits))
        ) / e.episode_length

        for i in range(e.episode_length):
            self.assertFalse(done)
            obs, rew, done, info = e.step(action)
            reward += rew

        self.assertTrue(done)
        self.assertTrue(np.abs(expected_value - reward) < 0.1)
