"""A game where the agent must output the suit of the card from k timesteps ago

The agent receives a sequence of cards, and must output the suit of the card
from k timesteps ago. E.g., if k == 1, then the agent must output the suit of
the previous card."""
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType

from popgym.core.deck import Deck
from popgym.core.env import POPGymEnv


class RepeatPrevious(POPGymEnv):
    """A game where the agent must output the suit of the card from k timesteps ago

    The agent receives a sequence of cards, and must output the suit of the card
    from k timesteps ago. E.g., if k == 1, then the agent must output the suit of
    the previous card.

    Args:
        num_decks: The number of decks to cycle through, which determines
            episode length
        k: The "previous" timestep. k == 0 denotes the current timestep, k==1
            denotes the previous timestep, ... k == 16 denotes 15 timesteps ago

    Returns:
        A gym environment
    """

    def __init__(self, num_decks=1, k=4):
        self.deck = Deck(num_decks)
        self.k = k
        self.max_episode_length = self.deck.num_cards - 1
        assert self.deck.num_cards > k, "k cannot be less than 52 * num_decks"
        self.deck.add_players("player")
        self.action_space = self.deck.get_obs_space(["suits"])
        self.observation_space = self.action_space
        self.state_space = gym.spaces.Tuple(
            (gym.spaces.MultiDiscrete([4] * k), gym.spaces.Box(0, 1, (4,)))
        )
        self.dealt_cards = np.zeros((4,), dtype=int)

    def get_state(self):
        cards = self.deck.show("player", ["suits_idx"])[0, -self.k :]
        if len(cards) != self.k:
            cards_ = np.zeros((self.k,), dtype=cards.dtype)
            cards_[-len(cards) :] = cards
            cards = cards_

        dealt_cards = 1.0 - self.dealt_cards / (self.deck.num_cards / 4)
        dealt_cards = dealt_cards.astype(np.float32)

        return cards.copy(), dealt_cards

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        reward_scale = 1 / (self.deck.num_cards - self.k)
        reward = 0

        terminated = len(self.deck) == 1

        if self.deck.hand_size("player") >= self.k:
            if action == self.deck.suits_idx[self.deck["player"][-self.k]]:
                reward = reward_scale
            else:
                reward = -reward_scale

        self.deck.deal("player", 1)
        card = self.deck.show("player", ["suits_idx"])[0, -1]
        self.dealt_cards[card] += 1
        obs = card.item()

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
        self.card = self.deck.show("player", ["suits_idx"])[0, -1]
        self.dealt_cards[:] = 0
        self.dealt_cards[self.card] += 1
        obs = self.card.item()
        return obs, {}


class RepeatPreviousEasy(RepeatPrevious):
    pass


class RepeatPreviousMedium(RepeatPrevious):
    def __init__(self):
        super().__init__(num_decks=2, k=32)


class RepeatPreviousHard(RepeatPrevious):
    def __init__(self):
        super().__init__(num_decks=3, k=64)
