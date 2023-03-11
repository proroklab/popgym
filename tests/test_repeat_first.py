from popgym.envs.repeat_first import RepeatFirst
from tests.base_env_test import AbstractTest


class TestRepeatFirst(AbstractTest.POPGymTest):
    def setUp(self) -> None:
        self.env = RepeatFirst()

    def test_perfect(self):
        terminated = truncated = False
        init_item, _ = self.env.reset()
        reward = 0
        for i in range(51):
            self.assertFalse(terminated or truncated)
            item, rew, terminated, truncated, info = self.env.step(init_item)
            self.assertTrue(item < 4)
            reward += rew

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(1.0, reward)
