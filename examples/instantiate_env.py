import gym

import popgym  # noqa: F401
from popgym.envs.higher_lower import HigherLower

# After import popgym
# You can either load them the normal way
env = HigherLower(num_decks=2)
obs = env.reset()

# or the gym way
env = gym.make("popgym-Blackjack-v0")
obs = env.reset()
