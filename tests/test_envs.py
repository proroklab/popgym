import pytest

from popgym import envs
from tests.test_utils import check_env_no_warnings


class TestEnvs:
    @pytest.mark.parametrize("env", envs.ALL)
    def test_no_warn(self, env):
        e = env()
        e.reset()
        check_env_no_warnings(e)
