import unittest

from popgym.envs.repeat_first import RepeatFirst
from tests.base_env_test import AbstractTest


class TestRepeatFirst(AbstractTest.POPGymTest):
    def setUp(self) -> None:
        self.env = RepeatFirst()

    def test_perfect(self):
        done = False
        init_item = self.env.reset()
        reward = 0
        for i in range(51):
            self.assertFalse(done)
            item, rew, done, info = self.env.step(init_item)
            self.assertTrue(item < 4)
            reward += rew

        self.assertTrue(done)
        self.assertAlmostEqual(1.0, reward)
