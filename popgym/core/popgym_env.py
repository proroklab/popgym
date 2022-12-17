from gym.core import ObsType

from gym import spaces, Env
from gym.core import ObsType

from popgym.util.definitions import Observability


class POPGymEnv(Env):
    state_space: spaces.Space[ObsType]
    observability_level: Observability = Observability.PARTIAL

    def get_state(self) -> ObsType:
        raise NotImplementedError
