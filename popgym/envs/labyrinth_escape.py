from typing import Any, Dict, Optional, Tuple, Union

import gym

from popgym.core.maze import Actions, Cell, MazeEnv


class LabyrinthEscape(MazeEnv):
    """A maze environment where the agent receives negative rewards
     until it finds the goal.

    Args:
        maze_dims: (heigh, width) of the generated mazes in blocks
        episode_length: maximum length of an episode

    Returns:
        A gym environment

    """

    def __init__(self, maze_dims=(10, 10), episode_length=1024):
        super().__init__(maze_dims, episode_length)
        self.neg_reward_scale = -1 / self.max_episode_length

    def step(self, action):
        super().step(action)
        reward = self.neg_reward_scale
        done = False
        y, x = self.position
        if self.maze.grid[y, x] == Cell.GOAL:
            reward += 1.0
            done = True
        done |= self.curr_step == self.max_episode_length - 1

        obs = self.get_obs(action)
        info = {"position": (x, y)}

        return obs, reward, done, info

    def render(self):
        print(self.tostring(start=True, end=True, agent=True, visited=True))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[gym.core.ObsType, Tuple[gym.core.ObsType, Dict[str, Any]]]:
        super().reset(seed=seed)
        # Based on free space
        x, y = self.position
        self.maze.grid[self.maze.end] = Cell.GOAL
        obs = self.get_obs(Actions.NONE)

        if return_info:
            return obs, {"maze": str(self.maze.grid)}
        return obs


class LabyrinthEscapeEasy(LabyrinthEscape):
    def __init__(self):
        super().__init__(maze_dims=(10, 10), episode_length=1024)


class LabyrinthEscapeMedium(LabyrinthEscape):
    def __init__(self):
        super().__init__(maze_dims=(14, 14), episode_length=1024)


class LabyrinthEscapeHard(LabyrinthEscape):
    def __init__(self):
        super().__init__(maze_dims=(18, 18), episode_length=1024)
