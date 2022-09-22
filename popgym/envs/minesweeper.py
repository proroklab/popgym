import enum
from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np


class HiddenSquare(enum.IntEnum):
    CLEAR = 0
    MINE = 1
    VIEWED = 2


class MineSweeper(gym.Env):
    """A game where the agent must press buttons in the reverse order it saw
    them pressed. E.g., seeing [1, 2, 3] means I should press them in the order
    [3, 2, 1].

    Args:
        difficulty: Easy, medium, or hard. Sets the board size and number of
            mines based on difficulty.

    Returns:
        A gym environment
    """

    def __init__(self, difficulty="easy"):
        assert difficulty in ["easy", "medium", "hard"]
        if difficulty == "easy":
            dims = 4, 4
            num_mines = 2
        elif difficulty == "medium":
            dims = 6, 6
            num_mines = 6
        elif difficulty == "hard":
            dims = 8, 8
            num_mines = 10
        else:
            raise NotImplementedError(f"Invalid difficulty {difficulty}")

        self.dims = dims
        self.num_mines = num_mines
        self.max_timesteps = dims[0] * dims[1] - num_mines
        self.success_reward_scale = 1 / self.max_timesteps
        self.fail_reward_scale = -0.5 - self.success_reward_scale
        # -1 for one step less (last action must be mine for lowest G)
        # -1 for one step less (first action is a view which will give reward)
        self.bad_action_reward_scale = -0.5 / (self.max_timesteps - 2)
        self.observation_space = gym.spaces.MultiDiscrete(
            np.array([9, *dims], dtype=np.int8)
        )
        self.action_space = gym.spaces.MultiDiscrete(np.array(dims))

    def step(self, action):
        done = False
        action = tuple(action)
        if self.hidden_grid[action] == HiddenSquare.MINE:
            done = True
            reward = self.fail_reward_scale
        elif self.hidden_grid[action] == HiddenSquare.VIEWED:
            # Querying already viewed square
            reward = self.bad_action_reward_scale
        else:
            self.hidden_grid[action] = HiddenSquare.VIEWED
            reward = self.success_reward_scale

        if self.timestep == self.max_timesteps:
            done = True

        if np.all(self.hidden_grid != HiddenSquare.CLEAR):
            # Uncovered all non-mine squares
            done = True

        obs = np.array([self.neighbor_grid[action], *action])
        self.timestep += 1

        return obs, reward, done, {}

    def render(self):
        visible_mask = self.hidden_grid == HiddenSquare.VIEWED
        result = np.full(self.hidden_grid.shape, ".")
        result[visible_mask] = self.neighbor_grid[visible_mask]
        out = " " + str(result).replace("[", "").replace("]", "").replace("'", "")
        print(out)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[gym.core.ObsType, Tuple[gym.core.ObsType, Dict[str, Any]]]:
        super().reset(seed=seed)
        # Init grids
        self.hidden_grid = np.full(self.dims, HiddenSquare.CLEAR, dtype=np.int8)
        mines_flat = np.random.choice(
            np.arange(self.dims[0] * self.dims[1]), size=self.num_mines, replace=False
        )
        self.hidden_grid.ravel()[mines_flat] = HiddenSquare.MINE
        self.neighbor_grid = np.zeros_like(self.hidden_grid)
        src_grid = np.pad(self.hidden_grid, [(1, 1), (1, 1)], constant_values=0)
        for shift_i in [-1, 0, 1]:
            for shift_j in [-1, 0, 1]:
                self.neighbor_grid += np.roll(
                    np.roll(src_grid, shift_i, 0), shift_j, 1
                )[1:-1, 1:-1]

        random_start = tuple(np.random.randint(self.dims))
        self.timestep = 0
        obs = np.array([self.neighbor_grid[random_start], *random_start])

        if return_info:
            return obs, {}

        return obs


if __name__ == "__main__":
    e = MineSweeper()
    obs = e.reset()
    e.render()
    done = False
    while not done:
        action = np.array(input("Enter x,y:").split(",")).astype(np.int8)
        obs, reward, done, info = e.step(action)
        e.render()
        print("reward", reward)


class MineSweeperEasy(MineSweeper):
    def __init__(self):
        super().__init__("easy")


class MineSweeperMedium(MineSweeper):
    def __init__(self):
        super().__init__("medium")


class MineSweeperHard(MineSweeper):
    def __init__(self):
        super().__init__("hard")
