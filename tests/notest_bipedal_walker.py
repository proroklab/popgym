import unittest

import numpy as np

from popgym.envs.bipedal_walker import BipedalWalker


class TestRepeatFirst(unittest.TestCase):
    def test_all(self):
        e = BipedalWalker()
        _ = e.reset()
        for i in range(100):
            _, _, done, _ = e.step(np.array([0.0, 0.0, 0.0, 0.0]))
            if done:
                e.reset()
