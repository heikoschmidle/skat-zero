import numpy as np

EPSILON = 0.2
ALPHA = 0.8


class Node():
    def __init__(self, state_id, state, allowed_actions, winner, game_type,
                 current_position, player_pos, cards, current_trick, track,
                 opp_one_cards, opp_two_cards):
        self.id = state_id
        self.state = state
        self.allowed_actions = allowed_actions
        self.winner = winner
        self.game_type = game_type
        self.current_position = current_position
        self.player_pos = player_pos
        self.cards = cards
        self.current_trick = current_trick
        self.track = track
        self.opp_one_cards = opp_one_cards
        self.opp_two_cards = opp_two_cards
        self.edges = []

    def is_leaf(self):
        if len(self.edges) > 0:
            return False
        return True


class Edge():
    def __init__(self, in_node, out_node, prior, action):
        self.id = in_node.id + '|' + out_node.id
        self.in_node = in_node
        self.out_node = out_node
        self.action = action
        self.stats = {
            'N': 0,
            'W': 0,
            'Q': 0,
            'P': prior,
        }


class MCTS():
    def __init__(self, root, cpuct):
        self.root = root
        self.tree = {}
        self.cpuct = cpuct
        self.add_node(root)

    def __len__(self):
        return len(self.tree)

    def move_to_leaf(self):
        breadcrumbs = []
        current_node = self.root
        done = 0
        value = 0

        while not current_node.is_leaf():
            maxQU = -99999

            if current_node == self.root:
                epsilon = EPSILON
                nu = np.random.dirichlet([ALPHA] * len(current_node.edges))
            else:
                epsilon = 0
                nu = [0] * len(current_node.edges)

            Nb = 0
            for action, edge in current_node.edges:
                Nb = Nb + edge.stats['N']

            for idx, (action, edge) in enumerate(current_node.edges):
                U = self.cpuct * \
                    ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx]) * \
                    np.sqrt(Nb) / (1 + edge.stats['N'])

                Q = edge.stats['Q']

                if Q + U > maxQU:
                    maxQU = Q + U
                    simulation_action = action
                    simulation_edge = edge

            _, value, done = current_node.state.take_action(simulation_action)
            current_node = simulation_edge.out_node
            breadcrumbs.append(simulation_edge)
        return current_node, value, done, breadcrumbs

    def back_fill(self, leaf, value, breadcrumbs):
            current_player = leaf.state.playerTurn
            for edge in breadcrumbs:
                player_turn = edge.player_turn
                if player_turn == current_player:
                    direction = 1
                else:
                    direction = -1

                edge.stats['N'] = edge.stats['N'] + 1
                edge.stats['W'] = edge.stats['W'] + value * direction
                edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

    def add_node(self, node):
        self.tree[node.id] = node
