import pytest
from gym.utils.env_checker import check_env

from popgym import ALL_ENVS
from popgym.wrappers.antialias import Antialias
from popgym.wrappers.previous_action import PreviousAction


@pytest.mark.parametrize("env", ALL_ENVS.keys())
def test_step_lastaction(env):
    wrapped_noaa = PreviousAction(env())
    wrapped_noaa.reset()
    check_env(wrapped_noaa)


@pytest.mark.parametrize("env", ALL_ENVS.keys())
def test_step_initialstep(env):
    wrapped_aa = Antialias(env())
    wrapped_aa.reset()
    check_env(wrapped_aa)


@pytest.mark.parametrize("env", ALL_ENVS.keys())
def test_step_lastaction_initialstep(env):
    wrapped_aa = Antialias(PreviousAction(env()))
    wrapped_aa.reset()
    check_env(wrapped_aa)
