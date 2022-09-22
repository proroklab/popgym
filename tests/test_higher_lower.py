import math
import unittest

from popgym.envs.higher_lower import HigherLower


class TestHigherLower(unittest.TestCase):
    def test_init(self):
        HigherLower()

    def test_episode_higher(self):
        env = HigherLower()
        done = False
        obs_list = [env.reset()]
        reward_list = []
        decklen = 52
        while not done:
            obs, reward, done, info = env.step(0)
            obs_list.append(obs)
            reward_list.append(reward)
        pairs = list(zip(obs_list[:-1], obs_list[1:]))
        pred_correct = [j > i for i, j in pairs]
        pred_incorrect = [j < i for i, j in pairs]
        pred_rew = (sum(pred_correct) - sum(pred_incorrect)) / decklen
        actual_rew = sum(reward_list)

        self.assertTrue(math.isclose(pred_rew, actual_rew))
