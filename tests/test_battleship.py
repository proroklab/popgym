import unittest

import numpy as np

from popgym.envs.battleship import Battleship


class TestBattleship(unittest.TestCase):
    def test_init_reset(self):
        b = Battleship()
        b.reset()

    def test_step(self):
        b = Battleship()
        action = (5, 9)
        assert b.action_space.contains(action)
        obs, reward, done, info = b.step(action)
        assert b.observation_space.contains(obs)

    def test_hit_miss(self):
        b = Battleship()
        ship_positions = np.where(b.board)  # (x_pos_arr, y_pos_arr)
        hit_pos = (ship_positions[0][0], ship_positions[1][0])
        no_ship_positions = np.where(~b.board.astype(bool))
        miss_pos = (no_ship_positions[0][0], no_ship_positions[1][0])
        obs1, reward1, done1, info1 = b.step(hit_pos)  # test hit
        assert obs1[0] == 1
        assert reward1 == 1.0 / sum(b.ship_sizes)
        obs2, reward2, done2, info2 = b.step(
            hit_pos
        )  # test duplicate hit position (should be a miss now)
        assert obs2[0] == 0
        assert reward2 == -1.0 / (b.max_episode_length - sum(b.ship_sizes))
        obs3, reward3, done3, info3 = b.step(miss_pos)  # test miss
        assert obs3[0] == 0
        assert reward3 == -1.0 / (b.max_episode_length - sum(b.ship_sizes))

    def test_num_ships(self):
        b = Battleship()
        assert sum(b.ship_sizes) == np.sum(b.board)

    def test_reward_sum(self):
        b = Battleship()

        # Episode reward 1.0 (only guesses hits)
        reward = 0.0
        ship_positions = np.where(b.board)  # (x_pos_arr, y_pos_arr)
        for i in range(len(ship_positions[0])):
            hit_pos = (ship_positions[0][i], ship_positions[1][i])
            obsi, rewardi, donei, infoi = b.step(hit_pos)
            reward += rewardi
            if i < len(ship_positions[0]) - 1:
                assert not donei
            elif i == len(ship_positions[0]) - 1:
                assert donei
        self.assertAlmostEqual(reward, 1.0)

        # Episode reward minimum
        b.reset()
        reward = 0.0
        no_ship_positions = np.where(~b.board.astype(bool))
        for i in range(b.max_episode_length):
            miss_pos = (
                no_ship_positions[0][i % len(no_ship_positions)],
                no_ship_positions[1][i % len(no_ship_positions)],
            )
            obsi, rewardi, donei, infoi = b.step(miss_pos)
            reward += rewardi
            if i < b.max_episode_length - 1:
                assert not donei
            elif i == b.max_episode_length - 1:
                assert donei
        self.assertAlmostEqual(
            reward, -1.0 * b.max_episode_length / len(no_ship_positions[0])
        )

        # Episode reward 0.0
        b.reset()
        reward = 0.0
        for x in range(b.board_size):
            for y in range(b.board_size):
                obsi, rewardi, donei, infoi = b.step((x, y))
                reward += rewardi
        self.assertAlmostEqual(reward, 0.0)
