import enum
from typing import Tuple, Optional

from gym import Env
from gym import spaces
from gym.core import ObsType, ActType, ObservationWrapper

from gym import Wrapper, spaces, Env
import numpy as np
from popgym.util.definitions import OBS, STATE, LAST_ACTION, ObservabilityLevel








class POPGymEnv(Env):
    state_space: spaces.Space[ObsType]
    observability_level: ObservabilityLevel = ObservabilityLevel.PARTIAL

    def get_state(self) -> ObsType:
        raise NotImplementedError
