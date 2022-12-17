import unittest

import numpy as np

from popgym.envs.minesweeper import HiddenSquare, MineSweeper
from tests.base_env_test import AbstractTest


class TestMineSweeper(AbstractTest.POPGymTest):
    def setUp(self) -> None:
        self.env = MineSweeper()

    def test_perfect(self):
        self.env.reset()
        xs, ys = np.where(self.env.hidden_grid == HiddenSquare.CLEAR)
        cum_rew = 0
        done = False
        for x, y in zip(xs, ys):
            self.assertFalse(done)
            obs, reward, done, info = self.env.step(np.array([x, y]))
            self.assertAlmostEqual(reward, self.env.success_reward_scale)
            cum_rew += reward

        self.assertTrue(done)
        self.assertAlmostEqual(cum_rew, 1.0)

    def test_worst(self):
        self.env.reset()
        ts = self.env.max_episode_length - 1
        xs, ys = np.where(self.env.hidden_grid == HiddenSquare.CLEAR)
        cum_rew = 0
        done = False
        for i in range(ts):
            action = np.array([xs[0], ys[0]])
            obs, reward, done, info = self.env.step(action)
            cum_rew += reward

        self.assertFalse(done)

        xs, ys = np.where(self.env.hidden_grid == HiddenSquare.MINE)
        action = np.array([xs[0], ys[0]])
        obs, reward, done, info = self.env.step(action)
        cum_rew += reward
        self.assertTrue(done)

        self.assertAlmostEqual(cum_rew, -1.0)

        cum_rew = 0
        for x, y in zip(xs, ys):
            obs, reward, done, info = self.env.step(np.array([x, y]))
            cum_rew += reward
