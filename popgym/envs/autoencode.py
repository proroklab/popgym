import enum
from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np

from popgym.core.deck import Deck
from popgym.core.env import POPGymEnv


class Mode(enum.IntEnum):
    PLAY = 0
    WATCH = 1


class Autoencode(POPGymEnv):
    """A game where the agent must press buttons in order it saw
    them pressed. E.g., seeing [1, 2, 3] means I should press them in the order
    [1, 2, 3].

    Args:
        num_decks: The maximum number of decks the agent must memorize

    Returns:
        A gym environment
    """

    def __init__(self, num_decks=1):
        self.deck = Deck(num_decks)
        self.deck.add_players("system")
        self.max_episode_length = self.deck.num_cards * 2 - 1
        self.action_space = self.deck.get_obs_space(["suits"])
        self.observation_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(2),
                self.action_space,
            )
        )
        self.state_space = gym.spaces.Tuple(
            (
                gym.spaces.MultiDiscrete([4] * self.deck.num_cards),
                gym.spaces.Discrete(2),
                gym.spaces.Box(0, 1, (1,)),
            )
        )
        self.mode = Mode.WATCH

    def make_obs(self, card_idx):
        card_suit = self.deck.suits_idx[card_idx].reshape(-1)
        return int(self.mode.value), card_suit.item()

    def get_state(self):
        cards = self.deck.suits_idx[self.deck.idx].copy()
        mode = int(self.mode.value)
        pos = np.array(
            [len(self.deck["system"]) / self.deck.num_cards], dtype=np.float32
        )
        return cards, mode, pos

    def step(self, action):
        done = False
        reward = 0
        # TODO: This is causing flaky tests, make sure we are not
        # off-by-one in step
        reward_scale = 1 / self.deck.num_cards

        if self.mode == Mode.WATCH:
            self.deck.deal("system", 1)
            if len(self.deck) == 0:
                self.mode = Mode.PLAY
                # Flip the cards so they play backwards
                # self.deck["system"].reverse()
            obs = self.make_obs(self.deck["system"][-1])
        else:
            # Recited all cards
            done = len(self.deck["system"]) == 1
            correct_card = self.deck.suits_idx[self.deck["system"].pop(-1)]
            if action == correct_card:
                reward = reward_scale
            else:
                reward = -reward_scale

            obs = self.make_obs(np.array([0]))

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
        self.deck.deal("system", 1)

        self.mode = Mode.WATCH
        obs = self.make_obs(self.deck["system"][-1])
        if return_info:
            return obs, {}

        return obs


class AutoencodeEasy(Autoencode):
    pass


class AutoencodeMedium(Autoencode):
    def __init__(self):
        super().__init__(num_decks=2)


class AutoencodeHard(Autoencode):
    def __init__(self):
        super().__init__(num_decks=3)
