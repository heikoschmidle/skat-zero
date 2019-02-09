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
        card = random.choice(possible_cards(cards, current_trick, game_type))

        return True, None, CARDS.index(card), None, None
