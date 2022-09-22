import enum
from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np

from popgym.core.deck import RANKS, Deck, DeckEmptyError


class Phase(enum.IntEnum):
    # The betting phase where the player selects a bet
    BET = 0
    # The cards are dealt to the player and house, player does not act
    DEAL = 1
    # Player chooses to hit or stay
    PLAY = 2
    # Player receives the final cards (for counting) and the reward
    PAYOUT = 3


def hand_value(hand):
    card_map = {
        "a": 1,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "10": 10,
        "j": 10,
        "q": 10,
        "k": 10,
    }
    value = 0
    has_ace = False
    for rank in hand:
        if rank == "a":
            has_ace = True
        value += card_map[rank]

    if value < 11 and has_ace:
        # ace = 1 + 10
        value += 10
    return value


class BlackJack(gym.Env):
    """A game of blackjack, where card counting is possible. Successful agents
    should learn to count cards, and bet higher/hit less often when the deck
    contains higher cards. Note that splitting is not allowed.

    Args:
        bet_sizes: The bet sizes available to the agent. These correspond to
            the final reward.
        num_decks: The number of individual decks combined into a single deck.
            In Vegas, this is usually between four and eight.
        max_rounds: The maximum number of rounds where the agent and dealer
            can hit/stay. There is no max in real blackjack, however this
            would result in a very large observation space.
        games_per_episode: The number of games per episode. This must be set high
            for card-counting to have an effect. When set to one, the game
            becomes fully observable.

    Returns:
        A gym environment
    """

    def __init__(
        self,
        bet_sizes=[0, 1 / 80, 1 / 40, 1 / 20],
        num_decks=1,
        max_rounds=6,
        games_per_episode=20,
    ):
        self.bet_sizes = bet_sizes
        self.max_rounds = max_rounds

        self.deck = Deck(num_decks=num_decks)
        self.deck.define_hand_value(hand_value, ["ranks"])
        self.deck.add_players("dealer", "player")

        # Hit, stay, and bet amount
        self.action_space = gym.spaces.Dict(
            {
                "hit": gym.spaces.Discrete(2),
                "bet_size": gym.spaces.Discrete(len(bet_sizes)),
            }
        )
        self.games_per_episode = games_per_episode
        self.curr_game = 0
        self.curr_round = 0
        self.observation_space = gym.spaces.Dict(
            {
                "phase": gym.spaces.Discrete(3),
                "dealer_hand": gym.spaces.MultiDiscrete(max_rounds * [RANKS.size]),
                "dealer_hand_cards_in_play": gym.spaces.MultiBinary(max_rounds),
                "player_hand": gym.spaces.MultiDiscrete(max_rounds * [RANKS.size]),
                "player_hand_cards_in_play": gym.spaces.MultiBinary(max_rounds),
            }
        )
        self.action_phase = Phase.BET

    def bet(self, action):
        # In bet action mode, we don't do anything
        # we inform the player to place a bet
        # which we set during the deal phase
        result = f"placed bet of {self.curr_bet}"
        return result

    def deal(self, action):
        # First round, serve the cards
        # TODO: Do not take bet twice...
        self.curr_bet = self.bet_sizes[action["bet_size"]]
        self.deck.deal("player", 2)
        self.deck.deal("dealer", 1)

    def play(self, action):
        player_hit = action["hit"]
        if player_hit:
            # Hit
            self.deck.deal("player", 1)
            result = "player hits"

        player_value = self.deck.value("player")
        player_bust = player_value > 21
        player_blackjack = player_value == 21
        player_natural = player_blackjack and self.deck.hand_size("player") == 2
        player_max_cards = self.deck.hand_size("player") == self.max_rounds

        game_done = (
            not player_hit or player_bust or player_blackjack or player_max_cards
        )
        if not game_done:
            return 0, game_done, result

        dealer_plays = (not player_bust and not player_hit) or player_blackjack

        if dealer_plays:
            dealer_value = self.deck.value("dealer")
            while dealer_value < 17 and self.deck.hand_size("dealer") < self.max_rounds:
                self.deck.deal("dealer", 1)
                dealer_value = self.deck.value("dealer")
        else:
            dealer_value = self.deck.value("dealer")

        dealer_bust = dealer_value > 21
        dealer_blackjack = dealer_value == 21
        dealer_natural = dealer_blackjack and self.deck.hand_size("dealer") == 2
        player_adv = player_value - dealer_value

        # compare
        if player_adv == 0:
            reward = 0
            result = f"player ({player_value}) and dealer ({dealer_value}) push"
        elif player_adv > 0:
            reward = self.curr_bet
            result = f"player ({player_value}) beats dealer ({dealer_value})"
        elif player_adv < 0:
            reward = -self.curr_bet
            result = f"player ({player_value}) loses to dealer ({dealer_value})"

        # busts
        if player_bust and not dealer_bust:
            reward = -self.curr_bet
            result = f"player ({player_value}) bust"
        elif dealer_bust and not player_bust:
            reward = self.curr_bet
            result = f"dealer ({dealer_value}) bust"
        elif dealer_bust and player_bust:
            reward = 0
            result = f"player ({player_value}) and dealer ({dealer_value}) bust"

        # naturals
        if player_natural and not dealer_natural:
            reward = 1.5 * self.curr_bet
            result = "player natural"
        elif dealer_natural and not player_natural:
            reward = -self.curr_bet
            result = "dealer natural"
        elif dealer_natural and player_natural:
            result = "push: player and dealer naturals"
            reward = 0

        return reward, game_done, result

    def game_reset(self):
        """Resets a game, but not the entire env."""
        self.curr_game += 1
        self.curr_round = 0
        self.deck.discard_all()

    def step(self, action):
        reward = 0
        done = False
        result = ""

        if len(self.deck) < 3:
            done = True
            result = "deck empty, episode over"

        try:
            game_done = False
            if self.action_phase == Phase.BET:
                result = self.bet(action)
                self.obs, self.info = self.build_obs_infos(result)
                self.action_phase = Phase.DEAL
            elif self.action_phase == Phase.DEAL or self.action_phase == Phase.PAYOUT:
                self.deal(action)
                self.obs, self.info = self.build_obs_infos(result)
                self.action_phase = Phase.PLAY
            elif self.action_phase == Phase.PLAY:
                reward, game_done, result = self.play(action)
                if game_done:
                    self.action_phase = Phase.PAYOUT
                if reward != 0:
                    assert self.action_phase == Phase.PAYOUT
                self.obs, self.info = self.build_obs_infos(result)
        except DeckEmptyError:
            done = True
            self.info["result"] += ", deck empty, episode over"

        self.curr_round += 1

        if game_done:
            # Game done, reset and goes to bet phase
            self.game_reset()
            self.action_phase = Phase.BET

        if self.curr_game == self.games_per_episode - 1:
            done = True

        return self.obs, reward, done, self.info

    def render(self):
        phase = Phase(self.obs["phase"]).name
        print(f"Phase: {phase}")
        print(f"Current Bet: {self.info['current_bet']}")
        dealer = self.deck.visualize_idx(self.info["dealer_hand_idx"])
        player = self.deck.visualize_idx(self.info["player_hand_idx"])
        dealer_val = self.deck.value_idx(self.info["dealer_hand_idx"])
        player_val = self.deck.value_idx(self.info["player_hand_idx"])
        print(f"dealer hand (sum={dealer_val}):\n{dealer}")
        print(f"player hand (sum={player_val}):\n{player}")
        print(self.info["result"])
        print("_______________________________")

    def build_obs_infos(self, result=""):
        # Convert card ids to color, suit, rank
        dealer = self.deck.show("dealer", ["ranks_idx"], pad_to=self.max_rounds)[0]
        dealer_size = self.deck.hand_size("dealer")
        dealer_hand_cards_in_play = np.zeros(self.max_rounds, dtype=np.int8)
        dealer_hand_cards_in_play[:dealer_size] = 1

        # Convert card ids to color, suit, rank
        player = self.deck.show("player", ["ranks_idx"], pad_to=self.max_rounds)[0]
        player_size = self.deck.hand_size("player")
        player_hand_cards_in_play = np.zeros(self.max_rounds, dtype=np.int8)
        player_hand_cards_in_play[:player_size] = 1

        obs: Dict[str, Any] = {
            "phase": self.action_phase.value,
            "dealer_hand": dealer.copy(),
            "dealer_hand_cards_in_play": dealer_hand_cards_in_play,
            "player_hand": player.copy(),
            "player_hand_cards_in_play": player_hand_cards_in_play,
        }
        infos: Dict[str, Any] = {
            "phase": self.action_phase,
            "current_bet": self.curr_bet,
            "dealer_hand_idx": self.deck.show("dealer", ["idx"])[0],
            "dealer_value": self.deck.value("dealer"),
            "player_hand_idx": self.deck.show("player", ["idx"])[0],
            "player_value": self.deck.value("player"),
            "result": result,
        }

        return obs, infos

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[gym.core.ObsType, Tuple[gym.core.ObsType, Dict[str, Any]]]:
        super().reset(seed=seed)
        self.deck.reset(rng=self.np_random)
        self.curr_game = 0
        self.curr_round = 0
        self.action_phase = Phase.BET
        self.curr_bet = -float("inf")
        self.obs, self.info = self.build_obs_infos()
        self.action_phase = Phase.DEAL
        if return_info:
            return self.obs, self.info

        return self.obs
