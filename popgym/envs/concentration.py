import math
from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np

from popgym.core.deck import Deck


class Concentration(gym.Env):
    """Classic game of concentration. A deck of cards is shuffled and placed
    face-down. The player can flip two cards, if they match they get a reward
    otherwise they dont.

    Args:
        num_decks: Number of decks to play with
        deck_type: String denoting what we are matching. Can be the card colors
        (colors) or the card ranks (ranks)

    Returns:
        A gym environment
    """

    def __init__(self, num_decks=1, deck_type="ranks"):
        # See https://math.stackexchange.com/questions/1876467/
        # how-many-turns-on-average-does-it-take-for-a-perfect-player
        # -to-win-concentrati
        n = 52 * num_decks
        self.episode_length = math.ceil(2 * n - (n / (2 * n - 1)))
        self.success_reward_scale = 1 / (52 * num_decks // 2)
        self.failure_reward_scale = -1 / (self.episode_length)

        self.deck = Deck(num_decks=num_decks)
        if deck_type == "colors":
            self.deck_type = self.deck.colors
            self.deck_idx_type = self.deck.colors_idx
        # Cannot do suits because there are 13 of each suit -- we need
        # an even number to match
        elif deck_type == "ranks":
            self.deck_type = self.deck.ranks
            self.deck_idx_type = self.deck.ranks_idx
        else:
            raise NotImplementedError(f"Invalid deck type {deck_type}")

        # cards = 14 * np.ones(len(self.deck))
        self.facedown_card = len(self.deck_type)
        cards = (1 + self.facedown_card) * np.ones(len(self.deck))
        self.observation_space = gym.spaces.MultiDiscrete(cards)
        self.action_space = gym.spaces.Discrete(np.array(len(self.deck)))
        self.deck.add_players("face_up", "face_up_idx", "in_play", "in_play_idx")

    def step(self, action):
        reward = 0
        done = False

        # Done conditions
        if self.curr_step >= self.episode_length - 1:
            done = True

        # Flip card
        self.deck["in_play"].append(self.deck.idx[action])
        self.deck["in_play_idx"].append(action)

        assert len(self.deck["in_play"]) <= 2
        # Cannot flip the same card twice
        self.obs = self.get_obs()

        assert len(self.deck["face_up"]) == len(self.deck["face_up_idx"])
        assert len(self.deck["in_play"]) == len(self.deck["in_play_idx"])

        # End of phase
        if len(self.deck["in_play"]) == 2:
            inplay_cards = self.deck_type[self.deck["in_play"]]
            cards_match = inplay_cards[0] == inplay_cards[1]
            flipped_same_idx = self.deck["in_play"][0] == self.deck["in_play"][1]
            if cards_match and not flipped_same_idx:
                reward = self.success_reward_scale
                self.deck["face_up"].extend(self.deck["in_play"])
                self.deck["face_up_idx"].extend(self.deck["in_play_idx"])
                if len(self.deck["face_up"]) == len(self.deck):
                    done = True
            else:
                reward = 2 * self.failure_reward_scale

            # Flip two last flipped-up cards face down again
            self.deck.discard_hands("in_play", "in_play_idx")
            self.hand = []

        self.curr_step += 1

        return self.obs, reward, done, {}

    def render(self, mode="ascii"):
        self.obs.tolist()
        visible_mask = self.obs != self.facedown_card
        rend = np.full(self.obs.shape, "?")
        rend[visible_mask] = self.obs[visible_mask]
        rend = rend.reshape(4, 13)
        output = (
            " "
            + str(rend)
            .replace("[", "")
            .replace("]", "")
            .replace(",", "")
            .replace("'", "")
            # .replace("\n", "")
        )
        print(output)

    def get_obs(self):
        obs = self.facedown_card * np.ones(len(self.deck), dtype=np.int64)
        hand_idx = self.deck["face_up_idx"] + self.deck["in_play_idx"]
        deck_idx = self.deck.idx[hand_idx]
        obs[hand_idx] = self.deck_idx_type[deck_idx]
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[gym.core.ObsType, Tuple[gym.core.ObsType, Dict[str, Any]]]:
        super().reset(seed=seed)
        self.flip_next_turn = False
        self.deck.reset(rng=self.np_random)
        self.curr_step = 0
        self.turn_counter = 0
        self.flipped_counter = 0
        self.obs = self.get_obs()
        info: Dict[str, Any] = {}
        if return_info:
            return self.obs, info

        return self.obs


class ConcentrationEasy(Concentration):
    def __init__(self):
        super().__init__(num_decks=1, deck_type="colors")


class ConcentrationMedium(Concentration):
    def __init__(self):
        super().__init__(num_decks=2, deck_type="colors")


class ConcentrationHard(Concentration):
    def __init__(self):
        super().__init__(num_decks=1, deck_type="ranks")
