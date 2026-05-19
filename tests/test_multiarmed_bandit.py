import numpy as np

from popgym.envs.multiarmed_bandit import MultiarmedBandit
from tests.base_env_test import AbstractTest


class TestMultiarmedBandit(AbstractTest.POPGymTest):
    def setUp(self) -> None:
        self.env = MultiarmedBandit()

    def test_best(self):
        obs, info = self.env.reset()
        bandits = info["bandits"]
        action = np.argmax(bandits)
        best_bandit = float(np.max(bandits))
        terminated = truncated = False
        reward = 0
        expected_value = 2 * best_bandit - 1

        for i in range(self.env.max_episode_length):
            self.assertFalse(terminated or truncated)
            obs, rew, terminated, truncated, info = self.env.step(action)
            reward += rew

        self.assertTrue(truncated)
        self.assertFalse(terminated)
        # Reward is stochastic; compare against a 4-sigma bound for finite horizon.
        reward_std = 2 * np.sqrt(best_bandit * (1 - best_bandit) / self.env.max_episode_length)
        self.assertTrue(np.abs(expected_value - reward) < 4 * reward_std)
