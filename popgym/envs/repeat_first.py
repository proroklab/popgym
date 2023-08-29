"""A game where the agent must output the suit of the initial card

The agent receives an initial card and indicator that this is the initial
card. Then, the agent receives a sequence of cards, and must output the
initial card at each timestep to receive a reward."""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType

from popgym.core.deck import Deck
from popgym.core.env import POPGymEnv


class RepeatFirst(POPGymEnv):
    """A game where the agent must output the suit of the initial card

    The agent receives an initial card and indicator that this is the initial
    card. Then, the agent receives a sequence of cards, and must output the
    initial card at each timestep to receive a reward.

    Args:
        num_decks: The number of decks to cycle through, which determines
            episode length

    Returns:
        A gym environment
    """

    def __init__(self, num_decks=1):
        self.deck = Deck(num_decks)
        self.deck.add_players("player")
        self.max_episode_length = self.deck.num_cards - 1
        self.action_space = self.deck.get_obs_space(["suits"])
        self.observation_space = self.action_space
        self.state_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(4), gym.spaces.Box(0, 1, (4,)))
        )
        self.dealt_cards = np.zeros((4,), dtype=int)

    def get_state(self):
        dealt_cards = 1.0 - self.dealt_cards / (self.deck.num_cards / 4)
        dealt_cards = dealt_cards.astype(np.float32)
        return self.card.copy(), dealt_cards

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        reward_scale = 1 / (self.deck.num_cards - 1)
        if action == self.card:
            reward = reward_scale
        else:
            reward = -reward_scale

        terminated = len(self.deck) == 1

        self.deck.deal("player", 1)
        card = self.deck.show("player", ["suits_idx"])[0, -1]
        self.dealt_cards[card] += 1
        obs = card.item()
        self.deck.discard_all()

        info: dict = {}

        return obs, reward, terminated, False, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[gym.core.ObsType, Dict[str, Any]]:
        super().reset(seed=seed)
        self.deck.reset(rng=self.np_random)
        self.deck.deal("player", 1)
        self.dealt_cards[:] = 0
        self.card = self.deck.show("player", ["suits_idx"])[0, -1]
        self.dealt_cards[self.card] += 1
        obs = self.card.item()
        return obs, {}


class RepeatFirstEasy(RepeatFirst):
    def __init__(self):
        super().__init__(num_decks=1)


class RepeatFirstMedium(RepeatFirst):
    def __init__(self):
        super().__init__(num_decks=8)


class RepeatFirstHard(RepeatFirst):
    def __init__(self):
        super().__init__(num_decks=16)
