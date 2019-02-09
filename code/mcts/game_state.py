import numpy as np
import hashlib
import copy

from code.constants import sort_cards, encode_binary
from code.rules import possible_cards
from code.constants import CARDS, POINTS
from code.game.game import evaluate_trick, evaluate_game


class State:
    def __init__(self, winner, game_type, current_position, player_pos, all_cards, current_trick,
                 track, points):
        self.current_position = current_position
        self.current_trick = current_trick
        self.track = track
        self.winner = winner
        self.game_type = game_type
        self.player_pos = player_pos
        self.all_cards = all_cards
        self.points = points
        self.cards = all_cards[current_position]
        self.id = None
        self.state = None
        self.actions = None

    def transform_to_state(self):
        self.state = list()
        for i in range(3):
            if i == self.player_pos:
                self.state.append([1] * 32)
            else:
                self.state.append([0] * 32)

        for i in range(3):
            if i == self.current_position:
                self.state.append([1] * 32)
            else:
                self.state.append([0] * 32)
        cards = self.all_cards[self.current_position]
        for card in sort_cards(cards):
            self.state.append(encode_binary([card]))
        for card in range(10 - len(cards)):
            self.state.append([0] * 32)

        track = self.track[:-len(self.current_trick)]
        for card in track:
            self.state.append(encode_binary([card]))
        for card in range(29 - len(track)):
            self.state.append([0] * 32)

        for card in self.current_trick:
            self.state.append(encode_binary([card]))
        for card in range(3 - len(self.current_trick)):
            self.state.append([0] * 32)

        s = ('').join([str(i) for sublist in self.state for i in sublist])
        self.id = str(int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**8)
        self.actions = self.allowed_actions()

    def allowed_actions(self):
        allowed_cards = possible_cards(self.cards, self.current_trick, self.game_type)
        allowed_actions = np.zeros(32, dtype=bool)
        for c in allowed_cards:
            allowed_actions[CARDS.index(c)] = True
        return allowed_actions

def take_action(state, card):
    new_state = copy.deepcopy(state)
    new_state.current_trick.append(card)
    new_state.track.append(card)
    new_state.all_cards[state.current_position].remove(card)

    value = 0
    done = False
    if len(new_state.track) == 30:
        winners = evaluate_game(new_state.points, new_state.player_pos, new_state.game_type, None)
        value = sum([new_state.points[p] for p in winners])
        done = True
        print('WINNER', winners)
    elif len(new_state.current_trick) == 3:
        winner = evaluate_trick(new_state.winner, new_state.current_trick, new_state.game_type)
        for c in new_state.current_trick:
            if c[1] in POINTS:
                new_state.points[winner] += POINTS[c[1]]
        new_state.current_position = winner
        new_state.winner = winner
        # print(new_state.current_trick, winner, state.winner)
        new_state.current_trick = []
    else:
        new_state.current_position += 1
        if new_state.current_position > 2:
            new_state.current_position -= 3

    new_state.cards = new_state.all_cards[new_state.current_position]

    new_state.transform_to_state()

    return value, done, new_state
