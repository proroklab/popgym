from gym import Env, Wrapper, spaces

from popgym.core.env import POPGymEnv
from popgym.core.observability import OBS, STATE, Observability


class ObservabilityWrapper(Wrapper):
    """Wrapper that adds the last action to the observation.

    Args:
        env: The environment.
        observability_level: The observability level.

    Returns:
        A gym environment
    """

    def __init__(self, env: Env, observability_level: Observability):
        super(ObservabilityWrapper, self).__init__(env)
        assert isinstance(
            env.unwrapped, POPGymEnv
        ), "This wrapper is made for POPGymEnvs."
        self.observability_level = observability_level
        if observability_level == Observability.FULL_AND_PARTIAL:
            self.observation_space = spaces.Dict(
                {
                    OBS: self.env.observation_space,
                    STATE: self.state_space,
                }
            )
        elif observability_level == Observability.FULL:
            self.observation_space = self.state_space

    def reset(self, **kwargs):
        if kwargs.get("return_info", False):
            obs, info = self.env.reset(**kwargs)
            obs, info = self.add_state(obs, info)
            return obs, info
        else:
            obs = self.env.reset(**kwargs)
            obs, _ = self.add_state(obs, {})
            return obs

    def add_state(self, obs, info):
        if self.observability_level == Observability.FULL:
            obs = self.get_state()
        elif self.observability_level == Observability.FULL_AND_PARTIAL:
            obs = {OBS: obs, STATE: self.get_state()}
        elif self.observability_level == Observability.FULL_IN_INFO_DICT:
            info[STATE] = self.get_state()
        return obs, info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs, info = self.add_state(obs, info)
        return obs, reward, done, info
