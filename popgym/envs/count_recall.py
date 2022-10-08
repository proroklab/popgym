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

    def __init__(self, max_episode_length=100, error_clamp=1, deck_type="colors"):
        self.deck = Deck(num_decks=1)
        if deck_type == "colors":
            self.deck_type = self.deck.colors
            self.deck_idx_type = self.deck.colors_idx
        elif deck_type == "suits":
            self.deck_type = self.deck.suits
            self.deck_idx_type = self.deck.suits_idx
        elif deck_type == "ranks":
            self.deck_type = self.deck.ranks
            self.deck_idx_type = self.deck.ranks_idx
        else:
            raise NotImplementedError(f"Invalid deck type {deck_type}")

        self.num_distinct_cards = len(self.deck_type)
        # Space: [dealt card, card query]
        self.observation_space = gym.spaces.MultiDiscrete([self.num_distinct_cards] * 2)
        self.action_space = gym.spaces.Box(
            high=max_episode_length,
            low=0,
            # Should actually be int but RLlib has issues
            dtype=np.float32,
            shape=(1,),
        )
        self.error_clamp = error_clamp
        self.max_episode_length = max_episode_length

    def render(self):
        dealt = self.deck_type[self.dealt]
        queried = self.deck_type[self.query]
        print(f"Dealt {dealt}, recall {queried} count")

    def step(self, action):
        done = False
        action = action.item()
        self.timestep += 1

        self.prev_query = self.query
        prev_count = self.counts[self.prev_query]
        self.query = self.queries[self.timestep]
        self.dealt = self.values[self.timestep]
        self.counts[self.dealt] += 1

        error = abs(prev_count - action)
        clamped = min(error, self.error_clamp)
        reward_scale = 1.0 / self.max_episode_length
        reward = reward_scale * (1 - 2 * clamped / self.error_clamp)

        if self.timestep >= self.max_episode_length:
            done = True

        obs = np.array([self.dealt, self.query], dtype=np.int64)
        info = {"counts": self.counts}

        return obs, reward, done, info

    def sample_deck(self):
        return np.random.choice(self.deck_idx_type, size=self.max_episode_length + 1)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[gym.core.ObsType, Tuple[gym.core.ObsType, Dict[str, Any]]]:
        super().reset(seed=seed)
        self.deck.reset(rng=self.np_random)

        self.counts = {k: 0 for k in range(self.num_distinct_cards)}
        self.timestep = 0
        self.values = self.sample_deck()
        self.queries = self.sample_deck()

        self.dealt = self.values[self.timestep]
        self.prev_query = None
        self.query = self.queries[self.timestep]
        self.counts[self.dealt] += 1

        obs = np.array([self.dealt, self.query], dtype=np.int64)

        if return_info:
            return obs, {"counts": self.counts}

        return obs


class CountRecallEasy(CountRecall):
    pass


class CountRecallMedium(CountRecall):
    def __init__(self, *args, **kwargs):
        super().__init__(max_episode_length=200, deck_type="suits")


class CountRecallHard(CountRecall):
    def __init__(self, *args, **kwargs):
        super().__init__(max_episode_length=400, deck_type="ranks")
