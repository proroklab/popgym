import unittest

import numpy as np

from popgym.envs.multiarmed_bandit import MultiarmedBandit


class TestBandits(unittest.TestCase):
    def test_step(self):
        m = MultiarmedBandit()
        m.reset()
        for i in range(100):
            obs, reward, done, info = m.step(np.random.randint(10))
