import unittest

import numpy as np

from popgym.core import deck


class TestCore(unittest.TestCase):
    def test_deck_size(self):
        d = deck.Deck(num_decks=1)
        self.assertEqual(len(d.idx), 52)
        d = deck.Deck(num_decks=7)
        self.assertEqual(len(d.idx), 7 * 52)

    def test_deal_discard_reset(self):
        d = deck.Deck(shuffle=False)
        orig_deck = d.idx.copy()
        d.add_players("a", "b")
        d.deal("a", 3)
        d.deal("b", 2)
        self.assertEqual(d.hand_size("a"), 3)
        self.assertEqual(d.hand_size("b"), 2)
        self.assertEqual(len(d), 52 - 5)

        d.discard_hands("a", "b")
        self.assertEqual(len(d), 52 - 5)

        d.reset(shuffle=False)
        self.assertEqual(len(d), 52)
        self.assertTrue(np.all(orig_deck == d.idx))

    def test_draw_52(self):
        d = deck.Deck()
        orig_deck = d.idx.copy()
        d.add_players("a")
        d.deal("a", 52)
        self.assertTrue(np.all(orig_deck == d.show("a", ["idx"])[0]))

    def test_deal_seq(self):
        d = deck.Deck(shuffle=False)
        d.add_players("a")
        num = 10
        for i in range(num):
            d.deal("a", 1)

        target = d.idx[52 : 52 - num - 1 : -1].tolist()
        self.assertEqual(len(target), num)
        self.assertTrue(d["a"] == target)

    def test_value(self):
        def vf(x):
            cmap = {
                "a": 0,
                "1": 2,
                "2": 4,
                "3": 6,
                "4": 8,
                "5": 10,
                "6": 12,
                "7": 14,
                "8": 16,
                "9": 18,
                "10": 20,
                "j": 21,
                "q": 22,
                "k": 23,
            }
            total = 0
            for c in x:
                total += cmap[c]
            return total

        d = deck.Deck(shuffle=True)
        d.define_hand_value(vf, ["ranks"])
        d.add_players("a")
        d.deal("a", 52)
        value = d.value("a")
        target = vf(d.ranks[d["a"]])
        self.assertEqual(target, value)
