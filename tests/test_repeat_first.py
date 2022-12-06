import unittest

from popgym.envs.repeat_first import RepeatFirst
from tests.base_env_test import AbstractTest


class TestRepeatFirst(AbstractTest.POPGymTest):
    def setUp(self) -> None:
        self.env = RepeatFirst()

    def test_perfect(self):
        done = False
        init_obs = self.env.reset()
        is_start, init_item = init_obs
        self.assertEqual(is_start, 1)
        reward = 0
        for i in range(51):
            self.assertFalse(done)
            obs, rew, done, info = self.env.step(init_item)
            is_start, item = obs
            self.assertTrue(item < 4)
            self.assertEqual(is_start, 0)
            reward += rew

        self.assertTrue(done)
        self.assertAlmostEqual(1.0, reward)
