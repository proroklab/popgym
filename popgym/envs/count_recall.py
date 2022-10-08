from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np

from popgym.core.deck import Deck


class CountRecall(gym.Env):
    """A game where the agent is queried on how many times it has observed
    an event in the past. This tests long-term order-agnostic memory like sets.

    Args:
        max_episode_length: The maximum number of timesteps in an episode
        error_clamp: Denotes the domain of the linear portion of the reward
            function. E.g. error_clamp == 2 means the agent will receive
            linearly-decreasing rewards for counts off by up to 2. Errors greater
            than 2 provide the same negative reward as an error of 2. Note as
            error_clamp -> inf, a crappy agent and perfect agent will acheive
            the same total reward.
        deck_type: What we use to count/differentiate cards.
            Can be colors, suits, or ranks.

    Returns:
        A gym environment
    """

    def __init__(self, num_decks=1, deck_type="colors"):
        self.value_deck = Deck(num_decks=num_decks)
        self.query_deck = Deck(num_decks=num_decks)
        self.value_deck.add_players("in_play")
        self.query_deck.add_players("in_play")
        if deck_type == "colors":
            self.deck_type = self.value_deck.colors
            self.deck_idx_type = self.value_deck.colors_idx
        elif deck_type == "suits":
            self.deck_type = self.value_deck.suits
            self.deck_idx_type = self.value_deck.suits_idx
        elif deck_type == "ranks":
            self.deck_type = self.value_deck.ranks
            self.deck_idx_type = self.value_deck.ranks_idx
        else:
            raise NotImplementedError(f"Invalid deck type {deck_type}")

        self.num_distinct_cards = len(np.unique(self.deck_type))
        self.max_card_count = self.value_deck.num_cards / self.num_distinct_cards
        # Space: [dealt card, card query]
        self.observation_space = gym.spaces.MultiDiscrete([self.num_distinct_cards] * 2)
        self.action_space = gym.spaces.Box(
            high=self.max_card_count,
            low=0,
            # Should actually be int but RLlib has issues
            dtype=np.float32,
            shape=(1,),
        )
        self.max_episode_length = self.value_deck.num_cards - 1

    def render(self):
        dealt = self.deck_type[self.value]
        queried = self.deck_type[self.query]
        print(f"Dealt {dealt}, recall {queried} count")

    def step(self, action):
        done = False
        action = action.item()

        self.prev_query = self.query
        prev_count = self.counts[self.prev_query]
        self.value, self.query = self.deal()
        self.counts[self.value] += 1

        # Error in [-1, 1]
        error = 2 * (0.5 - abs(prev_count - action) / self.max_card_count)
        reward_scale = 1.0 / self.max_episode_length
        reward = reward_scale * error

        if len(self.value_deck) == 0:  # self.timestep >= self.max_episode_length:
            done = True

        obs = np.array([self.value, self.query], dtype=np.int64)
        info = {"counts": self.counts}

        return obs, reward, done, info

    def sample_deck(self):
        return np.random.choice(self.deck_idx_type, size=self.max_episode_length + 1)

    def deal(self):
        self.value_deck.deal("in_play")
        self.query_deck.deal("in_play")
        return (
            self.deck_idx_type[self.value_deck["in_play"][-1]],
            self.deck_idx_type[self.query_deck["in_play"][-1]],
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[gym.core.ObsType, Tuple[gym.core.ObsType, Dict[str, Any]]]:
        super().reset(seed=seed)
        self.value_deck.reset(rng=self.np_random)
        # We dont want the decks to have the same order so reseed
        rng2, _ = gym.utils.seeding.np_random(self.np_random.randint(0, 1e15))
        self.query_deck.reset(rng=rng2)

        self.counts = {k: 0 for k in range(self.num_distinct_cards)}

        self.value, self.query = self.deal()
        self.prev_query = None
        self.counts[self.value] += 1

        obs = np.array([self.value, self.query], dtype=np.int64)

        if return_info:
            return obs, {"counts": self.counts}

        return obs


class CountRecallEasy(CountRecall):
    pass


class CountRecallMedium(CountRecall):
    def __init__(self, *args, **kwargs):
        super().__init__(num_decks=2, deck_type="suits")


class CountRecallHard(CountRecall):
    def __init__(self, *args, **kwargs):
        super().__init__(num_decks=4, deck_type="ranks")
