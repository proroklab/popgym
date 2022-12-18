import os
import unittest

from popgym.baselines import ppo


class TestPPO(unittest.TestCase):
    def test_ppo_mlp(self):
        os.environ["POPGYM_PROJECT"] = ""
        os.environ["POPGYM_GPU"] = "0"
        os.environ["POPGYM_STEPS"] = "80"
        os.environ["POPGYM_BPTT_CUTOFF"] = "10"
        os.environ["POPGYM_WORKERS"] = "1"
        os.environ["POPGYM_ENVS_PER_WORKER"] = "1"
        os.environ["POPGYM_MINIBATCH"] = "1"
        ppo.main()
