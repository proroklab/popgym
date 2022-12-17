import pytest
from gym.utils.env_checker import check_env

from popgym import ALL_ENVS
from popgym.wrappers.antialias_wrapper import AntialiasWrapper
from popgym.wrappers.last_action_wrapper import LastActionWrapper


@pytest.mark.parametrize("env", ALL_ENVS.keys())
def test_step_lastaction(env):
    wrapped_noaa = LastActionWrapper(env())
    wrapped_noaa.reset()
    check_env(wrapped_noaa)


@pytest.mark.parametrize("env", ALL_ENVS.keys())
def test_step_initialstep(env):
    wrapped_aa = AntialiasWrapper(env())
    wrapped_aa.reset()
    check_env(wrapped_aa)


@pytest.mark.parametrize("env", ALL_ENVS.keys())
def test_step_lastaction_initialstep(env):
    wrapped_aa = AntialiasWrapper(LastActionWrapper(env()))
    wrapped_aa.reset()
    check_env(wrapped_aa)
