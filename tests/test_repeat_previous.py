import unittest

from popgym.envs.repeat_previous import RepeatPrevious


class TestRepeatPrevious(unittest.TestCase):
    def test_all(self):
        e = RepeatPrevious()
        _ = e.reset()
        for i in range(100):
            _, _, done, _ = e.step(0)
            if done:
                e.reset()

    def test_k(self):
        e = RepeatPrevious(k=2)
        obs0 = e.reset()
        obs1, rew1, done1, info1 = e.step(0)
        self.assertFalse(done1)
        obs2, rew2, done2, info2 = e.step(obs0[1])
        self.assertFalse(done2)
        obs3, rew3, done3, info3 = e.step(obs1[1])
        self.assertFalse(done3)

    def test_full(self):
        e = RepeatPrevious(k=2)
        obs1 = e.reset()
        cum_rew = 0
        obs2, reward, done, info = e.step(0)
        cum_rew += reward

        all_obs = [obs1[1], obs2[1]]
        should_act = [obs1[0], obs2[0]]
        self.assertEqual(should_act[0], 0)
        self.assertEqual(should_act[1], 1)

        iters = 2
        while not done:
            a = all_obs[-2]
            obs, reward, done, info = e.step(a)
            self.assertEqual(obs[0], 1)
            all_obs.append(obs[1])
            cum_rew += reward
            iters += 1

        self.assertAlmostEqual(cum_rew, 1.0)

        self.assertEqual(iters, 52)

    def test_full_16(self):
        e = RepeatPrevious(k=16)
        all_obs = [e.reset()[1]]
        iters = 1
        cum_rew = 0
        while iters < 16:
            obs, reward, done, info = e.step(0)
            self.assertFalse(done)
            self.assertEqual(reward, 0)
            cum_rew += reward
            all_obs.append(obs[1])
            iters += 1

        while iters < 52:
            a = all_obs[-16]
            self.assertFalse(done)
            obs, reward, done, info = e.step(a)
            cum_rew += reward
            all_obs.append(obs[1])
            iters += 1

        self.assertTrue(done)
        self.assertAlmostEqual(cum_rew, 1.0)
