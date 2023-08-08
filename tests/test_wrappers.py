import pytest
from gymnasium.utils.env_checker import check_env

from popgym import envs
from popgym.core.observability import OBS, STATE, Observability
from popgym.wrappers.antialias import Antialias
from popgym.wrappers.discrete_action import DiscreteAction
from popgym.wrappers.flatten import Flatten
from popgym.wrappers.markovian import Markovian
from popgym.wrappers.previous_action import PreviousAction


def check_space(space, data):
    valid = space.contains(data)
    if not valid:
        raise ValueError(f"space {space} does not contain data {data}")


@pytest.mark.parametrize("env", envs.ALL.keys())
def test_previousaction_step(env):
    wrapped_noaa = PreviousAction(env())
    wrapped_noaa.reset()
    check_env(wrapped_noaa, skip_render_check=True)


@pytest.mark.parametrize("env", envs.ALL.keys())
def test_antialias_step(env):
    wrapped_aa = Antialias(env())
    wrapped_aa.reset()
    check_env(wrapped_aa, skip_render_check=True)


@pytest.mark.parametrize("env", envs.ALL.keys())
def test_previousaction_antialias_step(env):
    wrapped_aa = Antialias(PreviousAction(env()))
    wrapped_aa.reset()
    check_env(wrapped_aa, skip_render_check=True)


@pytest.mark.parametrize("env", envs.ALL.keys())
def test_markovian_state_space_full(env):
    wrapped = Markovian(env(), Observability.FULL)
    obs, _ = wrapped.reset()
    check_space(wrapped.observation_space, obs)
    check_space(wrapped.state_space, obs)
    for i in range(10):
        obs, reward, terminated, truncated, info = wrapped.step(
            wrapped.action_space.sample()
        )
        check_space(wrapped.observation_space, obs)
        check_space(wrapped.state_space, obs)
        if terminated or truncated:
            _ = wrapped.reset()


@pytest.mark.parametrize("env", envs.ALL.keys())
def test_markovian_state_space_partial(env):
    e = env()
    wrapped = Markovian(e, Observability.PARTIAL)
    obs, _ = wrapped.reset()
    check_space(wrapped.observation_space, obs)
    check_space(e.observation_space, obs)
    for i in range(10):
        obs, reward, terminated, truncated, info = wrapped.step(
            wrapped.action_space.sample()
        )
        check_space(wrapped.observation_space, obs)
        check_space(e.observation_space, obs)
        if terminated or truncated:
            _ = wrapped.reset()


@pytest.mark.parametrize("env", envs.ALL.keys())
def test_markovian_state_space_info_dict(env):
    e = env()
    wrapped = Markovian(e, Observability.FULL_IN_INFO_DICT)
    wrapped.reset()
    for i in range(10):
        obs, reward, terminated, truncated, info = wrapped.step(
            wrapped.action_space.sample()
        )
        check_space(wrapped.state_space, info[STATE])
        if terminated or truncated:
            _ = wrapped.reset()


@pytest.mark.parametrize("env", envs.ALL.keys())
def test_state_space_full_and_partial(env):
    e = env()
    wrapped = Markovian(e, Observability.FULL_AND_PARTIAL)
    obs, _ = wrapped.reset()
    check_space(wrapped.observation_space[STATE], obs[STATE])
    check_space(wrapped.observation_space[OBS], obs[OBS])
    check_space(e.observation_space, obs[OBS])
    for i in range(10):
        obs, reward, terminated, truncated, info = wrapped.step(
            wrapped.action_space.sample()
        )
        check_space(wrapped.observation_space[STATE], obs[STATE])
        check_space(wrapped.observation_space[OBS], obs[OBS])
        check_space(e.observation_space, obs[OBS])


@pytest.mark.parametrize("env", envs.ALL.keys())
def test_flatten_step(env):
    wrapped_aa = Flatten(env())
    obs, _ = wrapped_aa.reset()
    assert wrapped_aa.observation_space.contains(obs)
    check_env(wrapped_aa, skip_render_check=True)


@pytest.mark.parametrize("env", envs.ALL.keys())
def test_discrete_action(env):
    if issubclass(env, (envs.StatelessPendulum, envs.NoisyStatelessPendulum)):
        pytest.skip("StatelessPendulum does not support discrete action space")
    wrapped = DiscreteAction(Flatten(env()))
    _, _ = wrapped.reset()
    wrapped.step(wrapped.action_space.sample())
