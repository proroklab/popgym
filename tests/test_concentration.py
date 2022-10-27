import unittest

import numpy as np

from popgym.envs.concentration import Concentration


class TestConcentration(unittest.TestCase):
    def test_step(self):
        e = Concentration()
        _ = e.reset()
        for i in range(10000):
            _, _, done, _ = e.step(e.action_space.sample())
            if done:
                _ = e.reset()

    def test_repeatedly_flip_faceup(self):
        # Ensure we cannot match facedown cards
        # with previously-matched faceup cards
        env = Concentration()
        obs = env.reset()
        r_c = 0
        obs, r, d, _ = env.step(0)
        r_c += r
        cards = [obs[0]]
        found = False
        i = 0
        while not d and not found:
            i += 1
            obs, r, d, _ = env.step(i)
            r_c += r
            found = obs[i] in cards
            cards.append(obs[i])
        j = cards.index(obs[i])
        if not d and i % 2 != 1:
            obs, r, d, _ = env.step(j)
            r_c += r
        while not d:
            obs, r, d, _ = env.step(i)
            r_c += r
            i, j = j, i

        self.assertAlmostEqual(
            r_c,
            env.success_reward_scale
            + (env.episode_length - 2) * env.failure_reward_scale,
        )

    def test_perfect_game(self):
        e = Concentration()
        obs = e.reset()
        ranks = e.deck.ranks_idx[e.deck.idx]
        done = False
        running_reward = 0
        for i in range(13):
            idx = np.where(ranks == i)[0]
            for j in range(4):
                self.assertFalse(done)
                obs, reward, done, info = e.step(idx[j])
                self.assertLessEqual(len(e.deck["in_play"]), 2)
                running_reward += reward
                if j % 2 == 1:
                    self.assertEqual(reward, e.success_reward_scale)

        self.assertTrue(done)
        self.assertAlmostEqual(running_reward, 1.0)

    def test_two_decks(self):
        e = Concentration(num_decks=2)
        obs = e.reset()
        ranks = e.deck.ranks_idx[e.deck.idx]
        done = False
        running_reward = 0
        for i in range(13):
            idx = np.where(ranks == i)[0]
            for j in range(8):
                obs, reward, done, info = e.step(idx[j])
                running_reward += reward
                if j % 2 == 1:
                    self.assertEqual(reward, e.success_reward_scale)

        self.assertTrue(done)
        self.assertAlmostEqual(running_reward, 1.0)

    def test_perfect_game_colors(self):
        e = Concentration(deck_type="colors")
        obs = e.reset()
        colors = e.deck.colors_idx[e.deck.idx]
        done = False
        running_reward = 0
        for i in range(2):
            idx = np.where(colors == i)[0]
            for j in range(26):
                obs, reward, done, info = e.step(idx[j])
                running_reward += reward
                if j % 2 == 1:
                    print(e.deck["in_play"])
                    self.assertEqual(reward, e.success_reward_scale)

        self.assertTrue(done)
        self.assertAlmostEqual(running_reward, 1.0)

    def test_worst_game(self):
        e = Concentration()
        obs = e.reset()
        done = False
        running_reward = 0
        while not done:
            obs, reward, done, info = e.step(0)
            running_reward += reward

        self.assertAlmostEqual(running_reward, -1.0)
        self.assertTrue(done)

    def test_worst_game_colors(self):
        e = Concentration(deck_type="colors")
        obs = e.reset()
        done = False
        running_reward = 0
        while not done:
            obs, reward, done, info = e.step(0)
            running_reward += reward

        self.assertAlmostEqual(running_reward, -1.0)
        self.assertTrue(done)
