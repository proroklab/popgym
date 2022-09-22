from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np

from popgym.core.deck import SUITS, SUITS_UNICODE, Deck


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

    Returns:
        A gym environment
    """

    def __init__(self, num_decks=1, max_episode_length=100, error_clamp=2):
        self.deck = Deck(num_decks)
        self.action_space = self.deck.get_obs_space(["suits"])
        # Dealt card, query card
        self.observation_space = self.deck.get_obs_space(["suits", "suits"])
        self.error_clamp = error_clamp
        self.max_episode_length = max_episode_length
        self.action_space = gym.spaces.Box(
            low=0, high=max_episode_length, shape=(1,), dtype=np.float32
        )

    def render(self):
        dealt = SUITS_UNICODE[self.dealt]
        queried = SUITS_UNICODE[self.query]
        print(f"Dealt {dealt}, recall {queried} count")

    def step(self, action):
        done = False
        action = action.item()

        self.prev_query = self.query
        prev_count = self.counts[self.prev_query]
        self.query = self.sample_deck()
        self.dealt = self.sample_deck()
        self.counts[self.dealt] += 1

        error = abs(prev_count - action)
        clamped = min(error, self.error_clamp)
        reward_scale = 1.0 / self.max_episode_length
        reward = reward_scale * (1 - 2 * clamped / self.error_clamp)

        self.timestep += 1
        if self.timestep >= self.max_episode_length:
            done = True

        obs = np.array([self.dealt, self.query], dtype=np.float32)
        info = {"counts": self.counts}

        return obs, reward, done, info

    def sample_deck(self):
        return np.random.choice(self.deck.suits_idx)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[gym.core.ObsType, Tuple[gym.core.ObsType, Dict[str, Any]]]:
        super().reset(seed=seed)
        self.deck.reset(rng=self.np_random)

        self.counts = {k: 0 for k in range(len(SUITS))}
        self.timestep = 0
        self.dealt = self.sample_deck()
        self.prev_query = None
        self.query = self.sample_deck()
        self.counts[self.dealt] += 1

        obs = np.array([self.dealt, self.query], dtype=np.float32)

        if return_info:
            return obs, {"counts": self.counts}

        return obs
