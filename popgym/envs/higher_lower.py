from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np

from popgym.core.deck import Deck


def value_fn(hand):
    if hand[-1] > hand[-2]:
        return 1
    elif hand[-1] == hand[-2]:
        return 0
    else:
        return -1


class HigherLower(gym.Env):
    """A game of higher/lower. Given a deck of cards, the agent predicts whether the
    next card drawn from the deck is higher or lower than the last card drawn from
    the deck. A push results in zero reward, while a correct/incorrect guess result
    in 1/deck_size and -1/deck_size reward. The agent can learn to count cards to
    infer which cards are left in the deck, improving accuracy.

    Args:
        num_decks: The number of individual decks combined into a single deck.

    Returns:
        A gym environment
    """

    def __init__(self, num_decks=1):
        self.num_decks = num_decks
        self.deck = Deck(num_decks)
        self.deck.add_players("player")
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = self.deck.get_obs_space(["ranks"])
        self.value_map = dict(zip(self.deck.ranks, range(len(self.deck.ranks))))
        self.deck_size = len(self.deck)

    def step(self, action):
        guess_higher = action == 0
        if len(self.deck) <= 1:
            done = True
        else:
            done = False

        self.deck.deal("player", 1)
        assert self.deck.hand_size("player") == 2
        curr_idx, next_idx = self.deck["player"]
        curr_value, next_value = self.deck.ranks_idx[[curr_idx, next_idx]]

        rew_scale = 1 / self.deck_size
        if next_value == curr_value:
            reward = 0
        elif next_value > curr_value and guess_higher:
            reward = rew_scale
        elif next_value < curr_value and guess_higher:
            reward = -rew_scale
        elif next_value < curr_value and not guess_higher:
            reward = rew_scale
        elif next_value > curr_value and not guess_higher:
            reward = -rew_scale
        else:
            raise Exception("Should not reach this point")

        viz = np.stack(self.deck.show("player", ["suits", "ranks"])).T
        self.deck.discard("player", 0)
        obs = self.deck.show("player", ["ranks_idx"]).reshape(-1)

        return obs, reward, done, {"card": viz}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[gym.core.ObsType, Tuple[gym.core.ObsType, Dict[str, Any]]]:
        super().reset(seed=seed)
        self.deck.reset(rng=self.np_random)
        self.deck.deal("player", 1)
        obs = self.deck.show("player", ["ranks_idx"]).reshape(-1)
        viz = np.concatenate(self.deck.show("player", ["suits", "ranks"]))
        if return_info:
            return obs, {"card": viz}

        return obs


class HigherLowerEasy(HigherLower):
    def __init__(self):
        super().__init__(num_decks=1)


class HigherLowerMedium(HigherLower):
    def __init__(self):
        super().__init__(num_decks=2)


class HigherLowerHard(HigherLower):
    def __init__(self):
        super().__init__(num_decks=3)
