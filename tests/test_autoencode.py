import unittest

import numpy as np

from popgym.envs.autoencode import Mode, Autoencode


class TestAutoencode(unittest.TestCase):
    def test_init(self):
        Autoencode().reset()

    def test_step(self):
        b = Autoencode()
        obs = b.reset()
        done = False
        while not done:
            obs, reward, done, info = b.step(0)

    def test_full(self):
        b = Autoencode()
        obs = b.reset()
        done = False
        seq = [obs[1]]
        reward_sum = 0
        while obs[0] == Mode.WATCH:
            obs, reward, done, _ = b.step(0)
            seq.append(np.array(obs[1]))
            reward_sum += reward
        self.assertFalse(done)
        self.assertEqual(reward_sum, 0)
        for i in range(len(seq)):
            self.assertFalse(done)
            a = seq.pop(-1)
            _, reward, done, _ = b.step(a)
            reward_sum += reward
        self.assertTrue(done)
        self.assertEqual(len(b.deck["system"]), 0)

        self.assertAlmostEqual(reward_sum, 1.0)
