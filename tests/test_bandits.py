from popgym.envs.multiarmed_bandit import MultiarmedBandit
from tests.base_env_test import AbstractTest


class TestBandits(AbstractTest.POPGymTest):
    def setUp(self) -> None:
        self.env = MultiarmedBandit()
