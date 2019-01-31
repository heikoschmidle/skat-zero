import numpy as np


from code.mcts.game_state import take_action

EPSILON = 0.2
ALPHA = 0.8


class Node():
    def __init__(self, state, value, done):
        self.id = state.id
        self.current_position = state.current_position
        self.state = state
        self.allowed_actions = self.state.actions
        self.edges = []
        self.value = value
        self.done = done

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
            'N': 1.0,
            'W': 0.0,
            'Q': 0.0,
            'P': prior,
        }


class MCTS():
    def __init__(self, root, cpuct, epsilon, alpha):
        self.root = root
        self.tree = {}
        self.cpuct = cpuct
        self.epsilon = epsilon
        self.alpha = alpha
        self.add_node(root)

    def __len__(self):
        return len(self.tree)

    def move_to_leaf(self):
        breadcrumbs = []
        current_node = self.root

        i = 0
        while not current_node.is_leaf():
            maxQU = -99999.0
            i += 1

            if current_node == self.root:
                epsilon = self.epsilon
                nu = np.random.dirichlet([self.alpha] * len(current_node.edges))
            else:
                epsilon = 0.0
                nu = [0.0] * len(current_node.edges)

            Nb = 0.0
            for edge in current_node.edges:
                Nb = Nb + edge.stats['N']

            for idx, edge in enumerate(current_node.edges):
                if Nb < 0:
                    import ipdb; ipdb.set_trace()
                if edge.stats['N'] < 0:
                    import ipdb; ipdb.set_trace()

                U = self.cpuct * \
                    ((1.0 - epsilon) * edge.stats['P'] + epsilon * nu[idx]) * \
                    np.sqrt(Nb) / (1.0 + edge.stats['N'])

                Q = edge.stats['Q']

                # print(len(current_node.edges), U, Q, edge.stats, Nb)

                if U is np.nan:
                    import ipdb; ipdb.set_trace()

                if Q + U > maxQU:
                    maxQU = Q + U
                    simulation_edge = edge

            current_node = simulation_edge.out_node
            breadcrumbs.append(simulation_edge)

            if i > 1000:
                print(current_node)
                print(current_node.edges)
                print("Something went wrong")
                return None

        return current_node, breadcrumbs

    def back_fill(self, leaf, breadcrumbs, factor):
        current_player = leaf.current_position
        player_pos = leaf.state.player_pos
        for edge in breadcrumbs:
            player_turn = edge.in_node.current_position

            direction = 1.0
            if player_turn != player_pos and player_pos != current_player:
                direction = -1.0

            edge.stats['N'] = edge.stats['N'] + factor
            edge.stats['W'] = edge.stats['W'] + factor * leaf.value * direction
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

    def add_node(self, node):
        self.tree[node.id] = node
