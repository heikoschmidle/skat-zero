import numpy as np
import random
from itertools import cycle

from code.model.residual_nn import Residual_CNN
import code.mcts.mcts as mc
from code.mcts.game_state import State, take_action
from code.constants import sort_cards, encode_binary
from code.constants import CARDS
from code.rules import possible_cards
from code.game.game import evaluate_trick
from code.players.random_player import RandomPlayer


class ModelPlayer:
    def __init__(self, action_size, cpuct, mcts_simulations, config, model_file=None):
        self.action_size = action_size
        self.cpuct = cpuct
        self.mcts_simulations = mcts_simulations
        self.config = config
        self.model = self.create_model(model_file)
        self.mcts = None
        self.simulation_position = None
        self.current_trick = None
        self.current_position = None
        self.player_pos = None
        self.track = None
        self.game_type = None
        self.winner = None
        self.points = None

    def create_model(self, model_file):
        if model_file is not None:
            # Load from file
            print("Not implemented")
            return
        return Residual_CNN(self.config)

    def build_mcts(self, state):
        self.root = mc.Node(state, 0.0, False)
        self.mcts = mc.MCTS(self.root, self.cpuct)

    def change_root_mcts(self, state_id):
        self.mcts.root = self.mcts.tree[state_id]

    def simulate(self):
        # MOVE THE LEAF NODE
        leaf, breadcrumbs = self.mcts.move_to_leaf()

        import ipdb; ipdb.set_trace()

        # EVALUATE THE LEAF NODE
        self.evaluate_leaf(leaf)

        import ipdb; ipdb.set_trace()

        # BACKFILL THE VALUE THROUGH THE TREE
        self.mcts.back_fill(leaf, breadcrumbs)

        import ipdb; ipdb.set_trace()

    def opponents_card_estimate(self, track, cards, current_trick):
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

    def choose_card(self, winner, game_type, current_position, player_pos, cards,
                    current_trick, track, points, tau=1):

        self.start_position = current_position
        opp_one_cards, opp_two_cards = self.opponents_card_estimate(track, cards, current_trick)
        simulation_cards = {0: None, 1: None, 2: None}
        simulation_cards[current_position] = cards
        pool = cycle([0, 1, 2])
        for p in pool:
            if p == self.start_position:
                simulation_cards[next(pool)] = opp_one_cards
                simulation_cards[next(pool)] = opp_two_cards
                break

        self.points = points

        state = State(winner, game_type, current_position, player_pos, simulation_cards,
                      current_trick, track, points)

        state.transform_to_state()

        if self.mcts is None or state.id not in self.mcts.tree:
            self.build_mcts(state)
        else:
            self.change_root_mcts(state.id)

        for _ in range(10000):  #self.mcts_simulations):
            self.simulate()
            # import ipdb; ipdb.set_trace()

        # get action values
        pi, values = self.get_av(1)

        # pick the action
        action, value = self.chooseAction(pi, values, tau)

        next_state, _, _ = take_action(state, action)

        nn_value = -self.get_preds(next_state)[0]

        return (action, pi, value, nn_value)

    def get_preds(self, leaf):
        # predict the leaf
        input_to_model = np.array([np.reshape(leaf.state.state, (32, 48, 1))])

        preds = self.model.predict(input_to_model)
        value_array = preds[0]
        logits_array = preds[1]
        value = value_array[0]
        logits = logits_array[0]

        allowed_actions = leaf.allowed_actions
        logits[np.logical_not(allowed_actions)] = -100

        # SOFTMAX
        odds = np.exp(logits)
        probs = odds / np.sum(odds)

        return value, probs

    def evaluate_leaf(self, leaf):
        if leaf.done:
            return

        value, probs = self.get_preds(leaf)

        import ipdb; ipdb.set_trace()

        for idx, allowed in enumerate(leaf.allowed_actions):
            if allowed:
                new_value, new_done, new_state = take_action(leaf.state, CARDS[idx])
                if new_done:
                    print('FINISHED', new_value)
                    import ipdb; ipdb.set_trace()
                    leaf.done = new_done
                    leaf.value = new_value
                    return
                if new_state.id not in self.mcts.tree:
                    node = mc.Node(new_state, value[0], new_done)
                    self.mcts.add_node(node)
                    print(len(self.mcts.tree))
                else:
                    node = self.mcts.tree[new_state.id]
                    print('old')
                    # import ipdb; ipdb.set_trace()

                new_edge = mc.Edge(leaf, node, probs[idx], CARDS[idx])
                leaf.edges.append(new_edge)

    def get_av(self, tau):
        edges = self.mcts.root.edges
        pi = np.zeros(self.action_size, dtype=np.integer)
        values = np.zeros(self.action_size, dtype=np.float32)
        for action, edge in edges:
            pi[action] = pow(edge.stats['N'], 1/tau)
            values[action] = edge.stats['Q']

        pi = pi / (np.sum(pi) * 1.0)
        return pi, values

    def chooseAction(self, pi, values, tau):
        if tau == 0:
            actions = np.argwhere(pi == max(pi))
            action = random.choice(actions)[0]
        else:
            action_idx = np.random.multinomial(1, pi)
            action = np.where(action_idx == 1)[0][0]

        value = values[action]

        return action, value

    def replay(self, ltmemory):
        for i in range(config.TRAINING_LOOPS):
            minibatch = random.sample(ltmemory, min(config.BATCH_SIZE, len(ltmemory)))

            training_states = np.array([self.model.convertToModelInput(row['state']) for row in minibatch])
            training_targets = {
                'value_head': np.array([row['value'] for row in minibatch]),
                'policy_head': np.array([row['AV'] for row in minibatch])
            }

            self.model.fit(
                training_states,
                training_targets,
                epochs=config.EPOCHS,
                verbose=1,
                validation_split=0,
                batch_size=32
            )

    def predict(self, inputToModel):
        preds = self.model.predict(inputToModel)
        return preds
