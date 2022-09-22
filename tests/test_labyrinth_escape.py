import unittest

from popgym.envs.labyrinth_escape import LabyrinthEscape


class TestLabyrinthEscape(unittest.TestCase):
    def test_step(self):
        e = LabyrinthEscape()
        e.reset()
        done = False
        for i in range(1000):
            _, _, done, _ = e.step(e.action_space.sample())
            if done:
                e.reset()

    def test_goal(self):
        e = LabyrinthEscape((6, 6))
        done = False
        e.reset()
        for i in range(5):
            while not done:
                _, _, done, _ = e.step(e.action_space.sample())
            self.assertTrue(e.curr_step < e.episode_length)
            e.reset()

    def test_known(self):
        e = LabyrinthEscape((8, 8))
        e.reset(seed=1)
        left, right, _, down = 0, 1, 2, 3
        actions = [down] * 5 + [left] * 2 + [down] * 2 + [right] * 6 + [down] * 1
        done = False
        cum_rew = 0
        for action in actions:
            self.assertFalse(done)
            obs, reward, done, info = e.step(action)
            cum_rew += reward

        self.assertTrue(done)
        expected = 1.0 + len(actions) * e.neg_reward_scale
        self.assertTrue(done)
        self.assertAlmostEqual(cum_rew, expected)
