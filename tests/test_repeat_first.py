import unittest

from popgym.envs.repeat_first import RepeatFirst


class TestRepeatFirst(unittest.TestCase):
    def test_all(self):
        e = RepeatFirst()
        _ = e.reset()
        for i in range(100):
            _, _, done, _ = e.step(0)
            if done:
                e.reset()

    def test_perfect(self):
        e = RepeatFirst()
        done = False
        init_obs = e.reset()
        is_start, init_item = init_obs
        self.assertEqual(is_start, 1)
        reward = 0
        for i in range(51):
            self.assertFalse(done)
            obs, rew, done, info = e.step(init_item)
            is_start, item = obs
            self.assertTrue(item < 4)
            self.assertEqual(is_start, 0)
            reward += rew

        self.assertTrue(done)
        self.assertAlmostEqual(1.0, reward)
