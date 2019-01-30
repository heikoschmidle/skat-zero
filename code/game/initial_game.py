import json
import requests
from random import shuffle
import sys

from constants import CARDS, sort_cards

class Game:
    def __init__(self):
        self.cards = CARDS.copy()
        self.player = self.initialize_game()
        self.game = {
            "cards": self.cards,
            "player": self.player
        }

    def initialize_game(self):
        shuffle(self.cards)
        players_setup = []
        # skat = self.cards[:-2]
        for i in range(3):
            cards = self.cards[(i * 10):((i + 1) * 10)]
            game_estimate = self.reizen(cards, 0, i)
            players_setup.append(game_estimate)
        return self.choose_player(players_setup)

    def reizen(self, cards, first_position, position):
        cards = sort_cards(cards)
        payload = {
            "app": "Skat",
            "type": "reizValue",
            "cards": cards,
            "firstPosition": first_position,
            "position": position
        }
        reizwert = post_request(payload)

        payload = {
            "app": "Skat",
            "type": "shouldPlayHand",
            "cards": cards,
            "reizValue": reizwert["reizValue"],
            "firstPosition": first_position,
            "position": position
        }
        play_hand = post_request(payload)

        payload = {
            "app": "Skat",
            "type": "bestGame",
            "cards": cards,
            "hand": play_hand,
            "reizValue": reizwert["reizValue"],
            "firstPosition": first_position,
            "position": position
        }
        game = post_request(payload)
        game["reizwert"] = reizwert["reizValue"]
        game["play_hand"] = play_hand
        game["position"] = position
        return game

    def choose_player(self, setup):
        players = [eg for eg in setup if eg["play_hand"] is True]
        if len(players) == 0:
            return players
        return sorted(players, key=lambda k: k["reizwert"])[-1]

def post_request(payload):
    try:
        res = json.loads(
            requests.post(
                'http://127.0.0.1:10999/ai.json',
                json=payload).content.decode('utf-8')
        )
    except Exception as e:
        print(e)
        sys.exit(0)
    return res
