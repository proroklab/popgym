import unittest

import numpy as np

from popgym.envs.minesweeper import HiddenSquare, MineSweeper


class TestMineSweeper(unittest.TestCase):
    def test_step(self):
        e = MineSweeper()
        e.reset()
        done = False
        for i in range(1000):
            _, _, done, _ = e.step(e.action_space.sample())
            if done:
                e.reset()

    def test_perfect(self):
        e = MineSweeper()
        e.reset()
        xs, ys = np.where(e.hidden_grid == HiddenSquare.CLEAR)
        cum_rew = 0
        done = False
        for x, y in zip(xs, ys):
            self.assertFalse(done)
            obs, reward, done, info = e.step(np.array([x, y]))
            self.assertAlmostEqual(reward, e.success_reward_scale)
            cum_rew += reward

        self.assertTrue(done)
        self.assertAlmostEqual(cum_rew, 1.0)

    def test_worst(self):
        e = MineSweeper()
        e.reset()
        ts = e.max_timesteps - 1
        xs, ys = np.where(e.hidden_grid == HiddenSquare.CLEAR)
        cum_rew = 0
        for i in range(ts):
            action = np.array([xs[0], ys[0]])
            obs, reward, done, info = e.step(action)
            cum_rew += reward

        self.assertFalse(done)

        xs, ys = np.where(e.hidden_grid == HiddenSquare.MINE)
        action = np.array([xs[0], ys[0]])
        obs, reward, done, info = e.step(action)
        cum_rew += reward
        self.assertTrue(done)

        self.assertAlmostEqual(cum_rew, -1.0)

        cum_rew = 0
        for x, y in zip(xs, ys):
            obs, reward, done, info = e.step(np.array([x, y]))
            cum_rew += reward
