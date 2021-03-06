from code.constants import CARDS, NULL_CARDS, POINTS, SUIT_MAP, sort_cards

class Game:
    def __init__(self, memory, init_game, players):
        # print(init_game)
        self.memory = memory
        self.players = players
        self.cards = init_game["cards"]
        self.game_type = init_game["player"]["name"]
        self.player_pos = init_game["player"]["position"]
        self.points = {0: 0, 1: 0, 2: 0}

    def play(self):
        winner = 0
        track = []
        for trick_number in range(10):
            print('track', track)
            current_trick = []
            for i in range(3):
                current_position = winner + i
                if current_position > 2:
                    current_position -= 3
                player = self.players[current_position]
                cards_left = 10 - trick_number
                current_cards = self.cards[current_position * cards_left: (current_position + 1) * cards_left]
                if len(current_cards) > 1:
                    valid, state, card, pi, value = player.choose_card(
                        winner,
                        self.game_type,
                        current_position,
                        self.player_pos,
                        current_cards,
                        current_trick,
                        track,
                        self.points
                    )
                    if not valid:
                        import ipdb; ipdb.set_trace()
                        return None
                    current_trick.append(CARDS[card])
                    track.append(CARDS[card])
                    if player.type == 'model' and self.memory:
                        self.memory.commit_stmemory(state, value, pi, current_position)
                else:
                    card = current_cards[0]
                    current_trick.append(card)
                    track.append(card)

            winner = evaluate_trick(winner, current_trick, self.game_type)

            for c in current_trick:
                if c[1] in POINTS:
                    self.points[winner] += POINTS[c[1]]
                self.cards.remove(c)

        return evaluate_game(self.points, self.player_pos, self.game_type, self.memory)

def evaluate_trick(winner, current_trick, game_type):
    if game_type == "grand":
        local_winner = evaluate_grand_trick(current_trick)
    if game_type == "null":
        local_winner = evaluate_null_trick(current_trick)
    if game_type in ["karo", "herz", "pik", "kreuz"]:
        local_winner = evaluate_color_trick(current_trick, SUIT_MAP[game_type])
    winner += local_winner
    if winner > 2:
        winner -= 3
    return winner

def evaluate_grand_trick(current_trick):
    winner_card = current_trick[0]
    for card in current_trick[1:]:
        if card[1] == "U":
            if winner_card[1] == "U":
                if CARDS.index(card) < CARDS.index(winner_card):
                    winner_card = card
            else:
                winner_card = card
        else:
            if card[1] == "U":
                winner_card = card
            else:
                if card[0] == winner_card[0]:
                    if CARDS.index(card) < CARDS.index(winner_card):
                        winner_card = card

    return current_trick.index(winner_card)


def evaluate_null_trick(current_trick):
    winner_card = current_trick[0]
    for card in current_trick[1:]:
        if winner_card[0] == card[0]:
            if NULL_CARDS.index(card) < NULL_CARDS.index(winner_card):
                winner_card = card
    return current_trick.index(winner_card)

def evaluate_color_trick(current_trick, trump):
    winner_card = current_trick[0]
    for card in current_trick[1:]:
        if winner_card[1] == "U" or winner_card[0] == trump:
            if card[1] == "U" or card[0] == trump:
                if CARDS.index(card) < CARDS.index(winner_card):
                    winner_card = card
        else:
            if card[1] == "U" or card[0] == trump:
                winner_card = card
            else:
                if winner_card[0] == card[0]:
                    if CARDS.index(card) < CARDS.index(winner_card):
                        winner_card = card

    return current_trick.index(winner_card)


def evaluate_game(points, player_position, game_type, memory):
    idx = [0, 1, 2]
    winner = [player_position]
    if game_type == "null":
        if points[player_position] > 0:
            idx.remove(player_position)
            winner = idx
    else:
        if points[player_position] < 61:
            idx.remove(player_position)
            winner = idx

    if memory:
        for move in memory.stmemory:
            if move['current_pos'] in winner:
                move['value'] = 1  # points[move['current_pos']]
            else:
                move['value'] = -1  # points[move['current_pos']]
        memory.commit_ltmemory()

    print('Game winner', winner, memory)
    return winner
