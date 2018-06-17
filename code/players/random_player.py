import random
import requests
import json

from rules import possible_cards

class RandomPlayer:
    def __init__(self):
        pass

    def choose_card(self, current_pos, cards, current_trick, track, player_pos, game_type):
        return random.choice(possible_cards(cards, current_trick, game_type))
        
