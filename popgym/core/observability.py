import enum

OBS = "obs"
STATE = "state"


class Observability(enum.IntEnum):
    """Defines the observability level of the environment.

    To be used with popgym.wrappers.ObservabilityWrapper.

    - PARTIAL: Partial observation, the Markov state is not returned.
    - FULL_IN_INFO_DICT: Partial observation, the Markov state is provided
        by the info dict.
    - FULL: The Markov state is used as observation.
    - FULL_AND_PARTIAL: The observation is a dict {"obs": obs, "state": state}.
    """

    PARTIAL = 0
    FULL_IN_INFO_DICT = 1
    FULL = 2
    FULL_AND_PARTIAL = 3
