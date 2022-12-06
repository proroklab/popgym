from tests.base_env_test import AbstractTest


from popgym.envs.multiarmed_bandit import MultiarmedBandit


class TestBandits(AbstractTest.POPGymTest):
    def setUp(self) -> None:
        self.env = MultiarmedBandit()
