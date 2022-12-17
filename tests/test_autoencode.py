
import numpy as np

from popgym.envs.autoencode import Mode, Autoencode
from tests.base_env_test import AbstractTest


class TestAutoencode(AbstractTest.POPGymTest):
    def setUp(self) -> None:
        self.env = Autoencode()

    def test_full(self):
        obs = self.env.reset()
        done = False
        seq = [obs[1]]
        reward_sum = 0
        while obs[0] == Mode.WATCH:
            obs, reward, done, _ = self.env.step(0)
            seq.append(np.array(obs[1]))
            reward_sum += reward
        self.assertFalse(done)
        self.assertEqual(reward_sum, 0)
        for i in range(len(seq)):
            self.assertFalse(done)
            a = seq.pop(-1)
            _, reward, done, _ = self.env.step(a)
            reward_sum += reward
        self.assertTrue(done)
        self.assertEqual(len(self.env.deck["system"]), 0)

        self.assertAlmostEqual(reward_sum, 1.0)
