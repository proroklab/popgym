import unittest

import numpy as np

from popgym.envs.count_recall import CountRecall


class TestCountRecall(unittest.TestCase):
    def test_step(self):
        e = CountRecall()
        e.reset()
        done = False
        for i in range(1000):
            _, _, done, _ = e.step(e.action_space.sample())
            if done:
                e.reset()

    def test_perfect(self):
        e = CountRecall()
        counts = {k: 0 for k in range(e.num_distinct_cards)}

        obs, info = e.reset(return_info=True)
        d, q = obs
        counts[d] += 1
        action = np.array([counts[q]])
        reward = 0
        done = False

        t = 0
        while not done:
            obs, rew, done, info = e.step(action)
            d, q = obs
            counts[d] += 1
            action = np.array([counts[q]])
            self.assertEqual(rew, 1 / (e.value_deck.num_cards - 1))
            reward += rew
            t += 1

        self.assertEqual(t, e.max_episode_length)
        self.assertAlmostEqual(reward, 1.0)

    def test_awful(self):
        e = CountRecall()
        counts = {k: 0 for k in range(e.num_distinct_cards)}

        obs, info = e.reset(return_info=True)
        d, q = obs
        counts[d] += 1
        action = np.array([counts[q] + 2])
        reward = 0
        done = False

        t = 0
        while not done:
            obs, rew, done, info = e.step(action)
            d, q = obs
            counts[d] += 1
            action = np.array([counts[q] + 2])
            reward += rew
            t += 1

        self.assertEqual(t, e.max_episode_length)
        self.assertLess(reward, 1.0 - 0.1)
