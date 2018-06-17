from constants import SUIT_MAP


def possible_cards(cards, current_table, game_type):
    if len(current_table) == 0:
        return cards
    first_card = current_table[0]

    if game_type == "grand":
        return possible_grand_cards(cards, first_card)
    if game_type == "null":
        return possible_null_cards(cards, first_card)
    if game_type in ["karo", "herz", "pik", "kreuz"]:
        return possible_game_cards(cards, first_card, game_type)


def possible_grand_cards(cards, first_card):
    if first_card[1] == "U":
        return possible_unter_cards(cards)
    return possible_color_cards(cards, first_card)


def possible_null_cards(cards, first_card):
    return possible_color_cards(cards, first_card)


def possible_game_cards(cards, first_card, game_type):
    if first_card[1] == "U" or first_card[0] == SUIT_MAP[game_type]:
        return possible_trump_cards(cards, game_type)
    return possible_color_cards(cards, first_card)


def possible_color_cards(cards, first_card):
    possible_cards = [c for c in cards if c[0] == first_card[0] and c[1] != "U"]
    if len(possible_cards) > 0:
        return possible_cards
    return cards


def possible_unter_cards(cards):
    possible_cards = [c for c in cards if c[1] == "U"]
    if len(possible_cards) > 0:
        return possible_cards
    return cards


def possible_trump_cards(cards, game_type):
    possible_cards = [c for c in cards if c[1] == "U"]
    possible_cards += [c for c in cards if c[0] == SUIT_MAP[game_type] and c[1] != "U"]
    if len(possible_cards) > 0:
        return possible_cards
    return cards
