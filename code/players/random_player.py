import random
import requests
import json
from itertools import cycle

from code.rules import possible_cards
from code.players.opponents_estimate import random_estimate
from code.mcts.game_state import State
from code.constants import CARDS

class RandomPlayer:
    def __init__(self):
        self.type = 'random'

    def choose_card(self, winner, game_type, current_position, player_pos, cards, current_trick, track, points):
        # opp_one_cards, opp_two_cards = random_estimate(track, cards, current_trick)
        # simulation_cards = {0: None, 1: None, 2: None}
        # simulation_cards[current_position] = cards
        # pool = cycle([0, 1, 2])
        # for p in pool:
        #     if p == current_position:
        #         simulation_cards[next(pool)] = opp_one_cards
        #         simulation_cards[next(pool)] = opp_two_cards
        #         break

        # points = {}

        # state = State(winner, game_type, current_position, player_pos, simulation_cards,
        #               current_trick, track, points)

        # state.transform_to_state()
        card = random.choice(possible_cards(cards, current_trick, game_type))

        return None, CARDS.index(card), None, None
