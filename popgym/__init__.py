import inspect
from typing import Any, Dict, Union

import gym

from popgym.core.env import POPGymEnv
from popgym.envs.autoencode import (
    Autoencode,
    AutoencodeEasy,
    AutoencodeHard,
    AutoencodeMedium,
)
from popgym.envs.battleship import (
    Battleship,
    BattleshipEasy,
    BattleshipHard,
    BattleshipMedium,
)
from popgym.envs.concentration import (
    Concentration,
    ConcentrationEasy,
    ConcentrationHard,
    ConcentrationMedium,
)
from popgym.envs.count_recall import (
    CountRecall,
    CountRecallEasy,
    CountRecallHard,
    CountRecallMedium,
)
from popgym.envs.higher_lower import (
    HigherLower,
    HigherLowerEasy,
    HigherLowerHard,
    HigherLowerMedium,
)
from popgym.envs.labyrinth_escape import (
    LabyrinthEscape,
    LabyrinthEscapeEasy,
    LabyrinthEscapeHard,
    LabyrinthEscapeMedium,
)
from popgym.envs.labyrinth_explore import (
    LabyrinthExplore,
    LabyrinthExploreEasy,
    LabyrinthExploreHard,
    LabyrinthExploreMedium,
)
from popgym.envs.minesweeper import (
    MineSweeper,
    MineSweeperEasy,
    MineSweeperHard,
    MineSweeperMedium,
)
from popgym.envs.multiarmed_bandit import (
    MultiarmedBandit,
    MultiarmedBanditEasy,
    MultiarmedBanditHard,
    MultiarmedBanditMedium,
)
from popgym.envs.noisy_stateless_cartpole import (
    NoisyStatelessCartPole,
    NoisyStatelessCartPoleEasy,
    NoisyStatelessCartPoleHard,
    NoisyStatelessCartPoleMedium,
)
from popgym.envs.noisy_stateless_pendulum import (
    NoisyStatelessPendulum,
    NoisyStatelessPendulumEasy,
    NoisyStatelessPendulumHard,
    NoisyStatelessPendulumMedium,
)
from popgym.envs.repeat_first import (
    RepeatFirst,
    RepeatFirstEasy,
    RepeatFirstHard,
    RepeatFirstMedium,
)
from popgym.envs.repeat_previous import (
    RepeatPrevious,
    RepeatPreviousEasy,
    RepeatPreviousHard,
    RepeatPreviousMedium,
)
from popgym.envs.stateless_cartpole import (
    StatelessCartPole,
    StatelessCartPoleEasy,
    StatelessCartPoleHard,
    StatelessCartPoleMedium,
)
from popgym.envs.stateless_pendulum import (
    StatelessPendulum,
    StatelessPendulumEasy,
    StatelessPendulumHard,
    StatelessPendulumMedium,
)
from popgym.wrappers.antialias_wrapper import AntialiasWrapper
from popgym.wrappers.last_action_wrapper import LastActionWrapper


def wrap(env: Union[gym.Env, POPGymEnv]) -> Union[gym.Env, POPGymEnv]:
    """Wrap env using LastActionWrapper if necessary"""
    if isinstance(env, POPGymEnv) and env.obs_requires_prev_action:
        return AntialiasWrapper(LastActionWrapper(env))
    return env


#
# Simple envs
#
SIMPLE_ENVS: Dict[gym.Env, Dict[str, Any]] = {
    wrap(Autoencode): {"id": "popgym-Autoencode-v0"},
    wrap(RepeatPrevious): {"id": "popgym-RepeatPrevious-v0"},
    wrap(RepeatFirst): {"id": "popgym-RepeatFirst-v0"},
    wrap(CountRecall): {"id": "popgym-CountRecall-v0"},
}

SIMPLE_ENVS_EASY: Dict[gym.Env, Dict[str, Any]] = {
    wrap(AutoencodeEasy): {"id": "popgym-AutoencodeEasy-v0"},
    wrap(RepeatPreviousEasy): {"id": "popgym-RepeatPreviousEasy-v0"},
    wrap(RepeatFirstEasy): {"id": "popgym-RepeatFirstEasy-v0"},
    wrap(CountRecallEasy): {"id": "popgym-CountRecallEasy-v0"},
}

SIMPLE_ENVS_MEDIUM: Dict[gym.Env, Dict[str, Any]] = {
    wrap(AutoencodeMedium): {"id": "popgym-AutoencodeMedium-v0"},
    wrap(RepeatPreviousMedium): {"id": "popgym-RepeatPreviousMedium-v0"},
    wrap(RepeatFirstMedium): {"id": "popgym-RepeatFirstMedium-v0"},
    wrap(CountRecallMedium): {"id": "popgym-CountRecallMedium-v0"},
}

SIMPLE_ENVS_HARD: Dict[gym.Env, Dict[str, Any]] = {
    wrap(AutoencodeHard): {"id": "popgym-AutoencodeHard-v0"},
    wrap(RepeatPreviousHard): {"id": "popgym-RepeatPreviousHard-v0"},
    wrap(RepeatFirstHard): {"id": "popgym-RepeatFirstHard-v0"},
    wrap(CountRecallHard): {"id": "popgym-CountRecallHard-v0"},
}

ALL_SIMPLE_ENVS = {**SIMPLE_ENVS_EASY, **SIMPLE_ENVS_MEDIUM, **SIMPLE_ENVS_HARD}

#
# Control envs
#
CONTROL_ENVS: Dict[gym.Env, Dict[str, Any]] = {
    wrap(StatelessCartPole): {"id": "popgym-StatelessCartPole-v0"},
    wrap(StatelessPendulum): {
        "id": "popgym-StatelessPendulum-v0",
    },
    # BipedalWalker: {"id": "popgym-BipedalWalker-v0"},
}

CONTROL_ENVS_EASY: Dict[gym.Env, Dict[str, Any]] = {
    wrap(StatelessCartPoleEasy): {"id": "popgym-StatelessCartPoleEasy-v0"},
    wrap(StatelessPendulumEasy): {
        "id": "popgym-StatelessPendulumEasy-v0",
    },
    # BipedalWalkerEasy: {"id": "popgym-BipedalWalkerEasy-v0"},
}

CONTROL_ENVS_MEDIUM: Dict[gym.Env, Dict[str, Any]] = {
    wrap(StatelessCartPoleMedium): {"id": "popgym-StatelessCartPoleMedium-v0"},
    wrap(StatelessPendulumMedium): {
        "id": "popgym-StatelessPendulumMedium-v0",
    },
    # BipedalWalkerMedium: {"id": "popgym-BipedalWalkerMedium-v0"},
}

CONTROL_ENVS_HARD: Dict[gym.Env, Dict[str, Any]] = {
    wrap(StatelessCartPoleHard): {"id": "popgym-StatelessCartPoleHard-v0"},
    wrap(StatelessPendulumHard): {
        "id": "popgym-StatelessPendulumHard-v0",
    },
    # BipedalWalkerHard: {"id": "popgym-BipedalWalkerHard-v0"},
}

ALL_CONTROL_ENVS = {**CONTROL_ENVS_EASY, **CONTROL_ENVS_MEDIUM, **CONTROL_ENVS_HARD}

#
# Noisy envs
#
NOISY_ENVS: Dict[gym.Env, Dict[str, Any]] = {
    wrap(NoisyStatelessCartPole): {"id": "popgym-NoisyStatelessCartPole-v0"},
    wrap(NoisyStatelessPendulum): {
        "id": "popgym-NoisyStatelessPendulum-v0",
    },
    wrap(MultiarmedBandit): {"id": "popgym-MultiarmedBandit-v0"},
    # NonstationaryBandit: {"id": "popgym-NonstationaryBandit-v0"},
}

NOISY_ENVS_EASY: Dict[gym.Env, Dict[str, Any]] = {
    wrap(NoisyStatelessCartPoleEasy): {"id": "popgym-NoisyStatelessCartPoleEasy-v0"},
    wrap(NoisyStatelessPendulumEasy): {
        "id": "popgym-NoisyStatelessPendulumEasy-v0",
    },
    wrap(MultiarmedBanditEasy): {"id": "popgym-MultiarmedBanditEasy-v0"},
    # NonstationaryBanditEasy: {"id": "popgym-NonstationaryBanditEasy-v0"},
}

NOISY_ENVS_MEDIUM: Dict[gym.Env, Dict[str, Any]] = {
    wrap(NoisyStatelessCartPoleMedium): {
        "id": "popgym-NoisyStatelessCartPoleMedium-v0"
    },
    wrap(NoisyStatelessPendulumMedium): {
        "id": "popgym-NoisyStatelessPendulumMedium-v0",
    },
    wrap(MultiarmedBanditMedium): {"id": "popgym-MultiarmedBanditMedium-v0"},
    # NonstationaryBanditMedium: {"id": "popgym-NonstationaryBanditMedium-v0"},
}

NOISY_ENVS_HARD: Dict[gym.Env, Dict[str, Any]] = {
    wrap(NoisyStatelessCartPoleHard): {"id": "popgym-NoisyStatelessCartPoleHard-v0"},
    wrap(NoisyStatelessPendulumHard): {
        "id": "popgym-NoisyStatelessPendulumHard-v0",
    },
    wrap(MultiarmedBanditHard): {"id": "popgym-MultiarmedBanditHard-v0"},
    # NonstationaryBanditHard: {"id": "popgym-NonstationaryBanditHard-v0"},
}

ALL_NOISY_ENVS = {**NOISY_ENVS_EASY, **NOISY_ENVS_MEDIUM, **NOISY_ENVS_HARD}

#
# Game envs
#
GAME_ENVS: Dict[gym.Env, Dict[str, Any]] = {
    wrap(HigherLower): {"id": "popgym-HigherLower-v0"},
    wrap(Battleship): {"id": "popgym-Battleship-v0"},
    wrap(Concentration): {"id": "popgym-Concentration-v0"},
    wrap(MineSweeper): {"id": "popgym-MineSweeper-v0"},
}

GAME_ENVS_EASY: Dict[gym.Env, Dict[str, Any]] = {
    wrap(HigherLowerEasy): {"id": "popgym-HigherLowerEasy-v0"},
    wrap(BattleshipEasy): {"id": "popgym-BattleshipEasy-v0"},
    wrap(ConcentrationEasy): {"id": "popgym-ConcentrationEasy-v0"},
    wrap(MineSweeperEasy): {"id": "popgym-MineSweeperEasy-v0"},
}

GAME_ENVS_MEDIUM: Dict[gym.Env, Dict[str, Any]] = {
    wrap(HigherLowerMedium): {"id": "popgym-HigherLowerMedium-v0"},
    wrap(BattleshipMedium): {"id": "popgym-BattleshipMedium-v0"},
    wrap(ConcentrationMedium): {"id": "popgym-ConcentrationMedium-v0"},
    wrap(MineSweeperMedium): {"id": "popgym-MineSweeperMedium-v0"},
}

GAME_ENVS_HARD: Dict[gym.Env, Dict[str, Any]] = {
    wrap(HigherLowerHard): {"id": "popgym-HigherLowerHard-v0"},
    wrap(BattleshipHard): {"id": "popgym-BattleshipHard-v0"},
    wrap(ConcentrationHard): {"id": "popgym-ConcentrationHard-v0"},
    wrap(MineSweeperHard): {"id": "popgym-MineSweeperHard-v0"},
}

ALL_GAME_ENVS = {**GAME_ENVS_EASY, **GAME_ENVS_MEDIUM, **GAME_ENVS_HARD}

#
# Navigation envs
#
NAVIGATION_ENVS: Dict[gym.Env, Dict[str, Any]] = {
    wrap(LabyrinthExplore): {"id": "popgym-LabyrinthExplore-v0"},
    wrap(LabyrinthEscape): {"id": "popgym-LabyrinthEscape-v0"},
}

NAVIGATION_ENVS_EASY: Dict[gym.Env, Dict[str, Any]] = {
    wrap(LabyrinthExploreEasy): {"id": "popgym-LabyrinthExploreEasy-v0"},
    wrap(LabyrinthEscapeEasy): {"id": "popgym-LabyrinthEscapeEasy-v0"},
}

NAVIGATION_ENVS_MEDIUM: Dict[gym.Env, Dict[str, Any]] = {
    wrap(LabyrinthExploreMedium): {"id": "popgym-LabyrinthExploreMedium-v0"},
    wrap(LabyrinthEscapeMedium): {"id": "popgym-LabyrinthEscapeMedium-v0"},
}

NAVIGATION_ENVS_HARD: Dict[gym.Env, Dict[str, Any]] = {
    wrap(LabyrinthExploreHard): {"id": "popgym-LabyrinthExploreHard-v0"},
    wrap(LabyrinthEscapeHard): {"id": "popgym-LabyrinthEscapeHard-v0"},
}

ALL_NAVIGATION_ENVS = {
    **NAVIGATION_ENVS_EASY,
    **NAVIGATION_ENVS_MEDIUM,
    **NAVIGATION_ENVS_HARD,
}

#
# Sets of envs
#
ALL_BASE_ENVS: Dict[gym.Env, Dict[str, Any]] = {
    **SIMPLE_ENVS,
    **CONTROL_ENVS,
    **NOISY_ENVS,
    **GAME_ENVS,
    **NAVIGATION_ENVS,
}
ALL_EASY_ENVS: Dict[gym.Env, Dict[str, Any]] = {
    **SIMPLE_ENVS_EASY,
    **CONTROL_ENVS_EASY,
    **NOISY_ENVS_EASY,
    **GAME_ENVS_EASY,
    **NAVIGATION_ENVS_EASY,
}
ALL_MEDIUM_ENVS: Dict[gym.Env, Dict[str, Any]] = {
    **SIMPLE_ENVS_MEDIUM,
    **CONTROL_ENVS_MEDIUM,
    **NOISY_ENVS_MEDIUM,
    **GAME_ENVS_MEDIUM,
    **NAVIGATION_ENVS_MEDIUM,
}
ALL_HARD_ENVS: Dict[gym.Env, Dict[str, Any]] = {
    **SIMPLE_ENVS_HARD,
    **CONTROL_ENVS_HARD,
    **NOISY_ENVS_HARD,
    **GAME_ENVS_HARD,
    **NAVIGATION_ENVS_HARD,
}
ALL_ENVS: Dict[gym.Env, Dict[str, Any]] = {
    **ALL_EASY_ENVS,
    **ALL_MEDIUM_ENVS,
    **ALL_HARD_ENVS,
}

# Register envs
for e, v in ALL_ENVS.items():
    mod_name = inspect.getmodule(e).__name__  # type: ignore
    gym.envs.register(entry_point=":".join([mod_name, e.__name__]), **v)
