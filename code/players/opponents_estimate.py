import random

from code.constants import CARDS

def random_estimate(track, cards, current_trick):
    # Add here an estimate for the cards of the opponent
    random.seed(1)
    all_cards = CARDS.copy()
    for c in track + cards:
        all_cards.remove(c)
    random.shuffle(all_cards)

    if len(current_trick) == 0:
        num_cards_one = len(cards)
        num_cards_two = len(cards)
    if len(current_trick) == 1:
        num_cards_one = len(cards) - 1
        num_cards_two = len(cards)
    if len(current_trick) == 2:
        num_cards_one = len(cards) - 1
        num_cards_two = len(cards) - 1

    opp_one_cards = all_cards[:num_cards_one]
    opp_two_cards = all_cards[num_cards_one:(num_cards_one + num_cards_two)]

    return opp_one_cards, opp_two_cards
