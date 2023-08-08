from typing import Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType

from popgym.core.env import POPGymEnv
from popgym.core.wrapper import POPGymWrapper

PREV_ACTION = "prev_action"


class DiscreteAction(POPGymWrapper):
    """Wrapper that converts a MultiDiscrete into a single Discrete action.

    Args:
        env: The environment

    Returns:
        A gym environment
    """

    def __init__(self, env: POPGymEnv):
        super().__init__(env)
        self.action_space: spaces.Space
        if isinstance(self.action_space, spaces.Discrete):
            # Done, do nothing
            self.ravel_actions = False
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            self.ravel_actions = True
            self.old_action_space = self.action_space
            self.action_space = spaces.Discrete(np.prod(self.action_space.nvec))
        elif isinstance(self.action_space, (spaces.Tuple, spaces.Dict)):
            raise NotImplementedError(
                "Action space must be Discrete or MultiDiscrete, got a nested space"
                f" {self.action_space}.Please use the Flatten wrapper first."
            )
        else:
            raise NotImplementedError(
                "Action space must be Discrete or MultiDiscrete, got"
                f" {self.action_space}."
            )

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        if not self.ravel_actions:
            return self.env.step(action)

        discrete_action = np.unravel_index(action, self.old_action_space.nvec)
        obs, reward, terminated, truncated, info = self.env.step(discrete_action)
        return obs, reward, terminated, truncated, info
