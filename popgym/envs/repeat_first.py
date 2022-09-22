from typing import Any, Dict, Optional, Tuple, Union

import gym

from popgym.core.deck import Deck


class RepeatFirst(gym.Env):
    """A game where the agent must repeat the suit of the first card it saw

    Args:
        num_decks: The number of decks to cycle through, which determines
            episode length

    Returns:
        A gym environment
    """

    def __init__(self, num_decks=1):
        self.deck = Deck(num_decks)
        self.deck.add_players("player")
        self.action_space = self.deck.get_obs_space(["suits"])
        self.observation_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(2),
                self.action_space,
            )
        )

    def make_obs(self, card, is_start=False):
        return (int(is_start), card.reshape(-1))

    def step(self, action):
        done = False
        reward_scale = 1 / (self.deck.num_cards - 1)
        if action == self.card:
            reward = reward_scale
        else:
            reward = -reward_scale

        if len(self.deck) == 1:
            done = True

        self.deck.deal("player", 1)
        card = self.deck.show("player", ["suits_idx"])[0, -1]
        obs = self.make_obs(card)
        self.deck.discard_all()

        info = {}

        return obs, reward, done, info

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
        self.card = self.deck.show("player", ["suits_idx"])[0, -1]
        obs = self.make_obs(self.card, is_start=True)
        if return_info:
            return obs, {}

        return obs


class RepeatFirstEasy(RepeatFirst):
    def __init__(self):
        super().__init__(num_decks=1)


class RepeatFirstMedium(RepeatFirst):
    def __init__(self):
        super().__init__(num_decks=8)


class RepeatFirstHard(RepeatFirst):
    def __init__(self):
        super().__init__(num_decks=16)
