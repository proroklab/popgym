from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np


class Battleship(gym.Env):
    def __init__(self, board_size=10, ship_sizes=[2, 3, 3, 4]):
        # Params
        self.board_size = board_size
        self.ship_sizes = ship_sizes
        self.max_episode_length = self.board_size**2
        self.observation_space = gym.spaces.MultiDiscrete(
            np.array([2, self.board_size, self.board_size])
        )
        self.action_space = gym.spaces.MultiDiscrete([self.board_size, self.board_size])
        self.reset()

    def step(self, action):
        hit = False
        is_ship = self.board[action[0], action[1]]
        guessed_before = self.guesses[action[0], action[1]]
        self.guesses[action[0], action[1]] = 1
        if is_ship and not guessed_before:
            hit = True
            self.hits += 1

        self.num_steps += 1

        # Episode finishes when the max steps is reached
        # (which defaults to the number of squares on the board)
        #       or when all of the ships have been sunk
        done = (self.num_steps >= self.max_episode_length) or (
            self.hits == self.needed_hits
        )
        # Obs is 1 for a hit and 0 for a miss
        obs = np.concatenate(
            [
                np.array([int(hit)]),
                action,
            ]
        )
        # Reward is structured so that an episode of all hits will have a
        # total reward of 1.0,
        #
        # an episode with all misses will have a total reward of
        # -1.0 * (max_steps / (max_steps-num_hits_needed)),
        #
        # and an episode which simply guesses every single square on the
        # board will have a reward of 0.0
        reward = (int(hit) * (1.0 / self.needed_hits)) + (
            int(not hit) * (-1.0 / (self.max_episode_length - self.needed_hits))
        )
        info = {}

        return obs, reward, done, info

    def place_ship(self, board, ship_size):
        valid = False
        start = None
        end = None
        while not valid:
            idx_start = np.random.randint(self.board_size, size=2)
            direction = np.random.randint(2, size=2)
            idx_end = idx_start.copy()
            idx_end[direction[0]] = idx_start[direction[0]] + (-1) ** direction[1] * (
                ship_size - 1
            )
            if np.any(idx_end < 0) or np.any(idx_end >= self.board_size):
                continue
            start = np.minimum(idx_start, idx_end)
            end = np.maximum(idx_start, idx_end)
            valid = np.sum(board[start[0] : end[0] + 1, start[1] : end[1] + 1]) == 0
        board[start[0] : end[0] + 1, start[1] : end[1] + 1] = 1

        return board

    def render(self, mode=None):
        viz = np.chararray(self.board.shape, unicode=True)
        viz[self.board == 0] = "·"
        viz[self.board == 1] = "☐"
        viz[(self.guesses == 1) * (self.board == 1)] = "☒"
        viz[(self.guesses == 1) * (self.board == 0)] = "⚬"

        print(" " + str(viz).replace("[", "").replace("]", "").replace("'", ""))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[gym.core.ObsType, Tuple[gym.core.ObsType, Dict[str, Any]]]:

        if seed is not None:
            np.random.seed(seed)
        self.num_steps = 0

        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        for ship_size in self.ship_sizes:
            self.board = self.place_ship(self.board, ship_size)

        self.guesses = np.zeros((self.board_size, self.board_size), dtype=np.int8)

        self.hits = 0
        # Freebie
        free_idx = np.random.choice(np.where(self.board.ravel() == 0)[0])
        free = np.array(np.unravel_index(free_idx, self.board.shape))
        obs = np.concatenate([np.array([0]), free])

        self.needed_hits = sum(self.ship_sizes)
        if return_info:
            return obs, {}

        return obs


class BattleshipEasy(Battleship):
    def __init__(self):
        super().__init__(board_size=8, ship_sizes=[2, 3, 3, 4])


class BattleshipMedium(Battleship):
    def __init__(self):
        super().__init__(board_size=10, ship_sizes=[2, 3, 3, 4])


class BattleshipHard(Battleship):
    def __init__(self):
        super().__init__(board_size=12, ship_sizes=[2, 3, 3, 4])
