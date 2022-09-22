import unittest

from popgym.envs.blackjack import BlackJack


class TestBlackjack(unittest.TestCase):
    def test_reset(self):
        b = BlackJack()
        b.reset()

    def test_step(self):
        b = BlackJack()
        b.reset()
        a = {"hit": 1, "bet_size": 1}
        [b.step(a) for i in range(10)]

    def test_many_step(self):
        b = BlackJack()
        b.reset()
        a = {"hit": 1, "bet_size": 1}
        for i in range(2000):
            obs, reward, done, info = b.step(a)
            if done:
                b.reset()

    def test_render(self):
        b = BlackJack()
        b.reset()
        a = {"hit": 1, "bet_size": 1}
        [b.step(a) for i in range(10)]
        b.render()
