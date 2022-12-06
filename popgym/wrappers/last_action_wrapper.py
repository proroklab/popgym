from typing import Tuple, Optional

import numpy as np
from gym import Wrapper, spaces, Env
from gym.core import ObsType, ActType

from popgym.util.definitions import OBS, STATE, LAST_ACTION


class LastActionWrapper(Wrapper):
    def __init__(self, env: Env, null_action: Optional[ActType] = None):
        super().__init__(env)
        self.observation_space = LastActionWrapper.add_act_space_to_obs_space(self.env.observation_space,
                                                                              self.env.action_space)
        if null_action is None:
            null_action = LastActionWrapper.get_null_action(self.action_space)
        assert self.action_space.contains(null_action)
        self.null_action = null_action

    @staticmethod
    def add_act_space_to_obs_space(observation_space: spaces.Space, action_space: spaces.Space):
        if isinstance(observation_space, (spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary)):
            observation_space = spaces.Tuple((observation_space, action_space))
        elif isinstance(observation_space, spaces.Tuple):
            observation_space = spaces.Tuple(tuple(observation_space.spaces) + (action_space,))
        elif isinstance(observation_space, spaces.Dict):
            observation_space = observation_space.spaces.copy()
            if set(observation_space.keys()) == {OBS, STATE}:
                # Observation comes from ObservabilityWrapper with observability level FULL_AND_PARTIAL
                observation_space[OBS] = LastActionWrapper.add_act_space_to_obs_space(observation_space[OBS],
                                                                                      action_space)
            else:
                observation_space[LAST_ACTION] = action_space
            observation_space = spaces.Dict(observation_space)
        else:
            raise ValueError("Unknown observation space")
        return observation_space

    @staticmethod
    def add_act_to_obs(observation_space: spaces.Space, obs: ObsType, action: ActType):
        if isinstance(observation_space, (spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary)):
            obs = (obs, action)
        elif isinstance(observation_space, spaces.Tuple):
            obs = (*obs, action)
        elif isinstance(observation_space, spaces.Dict):
            if set(observation_space.keys()) == {OBS, STATE}:
                # Observation comes from ObservabilityWrapper with observability level FULL_AND_PARTIAL
                obs[OBS] = LastActionWrapper.add_act_to_obs(observation_space[OBS], obs[OBS], action)
            else:
                obs[LAST_ACTION] = action
        else:
            raise ValueError("Unknown observation space")
        return obs

    @staticmethod
    def get_null_action(action_space):
        if isinstance(action_space, (spaces.Discrete, spaces.MultiBinary, spaces.MultiDiscrete, spaces.Box)):
            action = np.zeros(action_space.shape, action_space.dtype)
            if not action_space.contains(action):
                action = action_space.low
        elif isinstance(action_space, spaces.Tuple):
            action = tuple(LastActionWrapper.get_null_action(action_space_) for action_space_ in action_space)
        elif isinstance(action_space, spaces.Dict):
            action = {key: LastActionWrapper.get_null_action(value) for key, value in action_space.items()}
        else:
            raise NotImplementedError
        return action

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        obs = LastActionWrapper.add_act_to_obs(self.observation_space, obs, action)
        return obs, reward, done, info

    def reset(self, **kwargs):
        if kwargs.get("return_info", False):
            obs, info = self.env.reset(**kwargs)
            obs = LastActionWrapper.add_act_to_obs(self.observation_space, obs, self.null_action)
            return obs, info
        else:
            obs = self.env.reset(**kwargs)
            obs = LastActionWrapper.add_act_to_obs(self.observation_space, obs, self.null_action)
            return obs
