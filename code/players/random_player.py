import random
import requests
import json

from code.rules import possible_cards

class RandomPlayer:
    def __init__(self):
        pass

    def choose_card(self, cards, current_trick, game_type):
        return random.choice(possible_cards(cards, current_trick, game_type))

