import unittest

import numpy as np

from popgym.envs.count_recall import CountRecall
from tests.base_env_test import AbstractTest


class TestCountRecall(AbstractTest.POPGymTest):
    def setUp(self) -> None:
        self.env = CountRecall()

    def test_perfect(self):
        counts = {k: 0 for k in range(self.env.num_distinct_cards)}

        obs, info = self.env.reset(return_info=True)
        d, q = obs
        counts[d] += 1
        action = np.array([counts[q]])
        reward = 0
        done = False

        t = 0
        while not done:
            obs, rew, done, info = self.env.step(action)
            d, q = obs
            counts[d] += 1
            action = np.array([counts[q]])
            self.assertEqual(rew, 1 / (self.env.value_deck.num_cards - 1))
            reward += rew
            t += 1

        self.assertEqual(t, self.env.max_episode_length)
        self.assertAlmostEqual(reward, 1.0)

    def test_awful(self):
        counts = {k: 0 for k in range(self.env.num_distinct_cards)}

        obs, info = self.env.reset(return_info=True)
        d, q = obs
        counts[d] += 1
        action = np.array([counts[q] + 2])
        reward = 0
        done = False

        t = 0
        while not done:
            obs, rew, done, info = self.env.step(action)
            d, q = obs
            counts[d] += 1
            action = np.array([counts[q] + 2])
            reward += rew
            t += 1

        self.assertEqual(t, self.env.max_episode_length)
        self.assertLess(reward, 1.0 - 0.1)
