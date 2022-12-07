import enum

OBS = "obs"
STATE = "state"
LAST_ACTION = "last_action"


class Observability(enum.IntEnum):
    PARTIAL = 0
    FULL_IN_INFO_DICT = 1
    FULL = 2
    FULL_AND_PARTIAL = 3
