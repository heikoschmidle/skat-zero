import numpy as np
import random
from itertools import cycle

from code.model.residual_nn import Residual_CNN
import code.mcts.mcts as mc
from code.mcts.game_state import State, take_action
from code.constants import sort_cards, encode_binary
from code.constants import CARDS
from code.cnn_setup import INPUT_DIMENSION
from code.rules import possible_cards
from code.game.game import evaluate_trick
from code.players.random_player import RandomPlayer
from code.players.opponents_estimate import random_estimate


class ModelPlayer:
    def __init__(self, config, model_file=None):
        self.action_size = config['ACTION_SIZE']
        self.cpuct = config['CPUCT']
        self.epsilon = config['CPUCT']
        self.alpha = config['ALPHA']
        self.mcts_simulations = config['MC_SIMULATIONS']
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
        self.type = 'model'

    def create_model(self, model_file):
        if model_file is not None:
            # Load from file
            print("Not implemented")
            return
        return Residual_CNN(self.config)

    def build_mcts(self, state):
        self.root = mc.Node(state, 0.0, False)
        self.mcts = mc.MCTS(self.root, self.cpuct, self.epsilon, self.alpha)

    def change_root_mcts(self, state_id):
        self.mcts.root = self.mcts.tree[state_id]

    def simulate(self, factor):
        # MOVE THE LEAF NODE
        # print('1')
        leaf, breadcrumbs = self.mcts.move_to_leaf()

        if leaf is None:
            return False

        # EVALUATE THE LEAF NODE
        # print('2')
        self.evaluate_leaf(leaf)

        # BACKFILL THE VALUE THROUGH THE TREE
        # print('3')
        self.mcts.back_fill(leaf, breadcrumbs, factor)

        return True

    def choose_card(self, winner, game_type, current_position, player_pos, cards,
                    current_trick, track, points, tau=0, factor=0.100):

        # print('Choosing card...')
        # print(cards)
        # print(player_pos, current_position, current_trick, track)

        self.start_position = current_position
        opp_one_cards, opp_two_cards = random_estimate(track, cards, current_trick)
        simulation_cards = {0: None, 1: None, 2: None}
        simulation_cards[current_position] = cards
        pool = cycle([0, 1, 2])
        for p in pool:
            if p == self.start_position:
                simulation_cards[next(pool)] = opp_two_cards
                simulation_cards[next(pool)] = opp_one_cards
                break
        # print(simulation_cards)
        self.points = points

        state = State(winner, game_type, current_position, player_pos, simulation_cards,
                      current_trick, track, points)

        state.transform_to_state()

        # import ipdb; ipdb.set_trace()

        if self.mcts is None or state.id not in self.mcts.tree:
            self.build_mcts(state)
        else:
            self.change_root_mcts(state.id)

        for i in range(self.mcts_simulations):
            # if i % 10 == 0:
            #     print(i)
            success = self.simulate(factor)
            if not success:
                return False

        # get action values
        pi, values = self.get_av(1)

        # pick the action
        action, value = self.choose_action(pi, values, tau)

        # _, _, next_state = take_action(state, CARDS[action])

        # nn_value = -self.get_preds(mc.Node(next_state, None, None))[0]

        return True, state, action, pi, value

    def get_preds(self, leaf):
        input_to_model = np.array([np.reshape(leaf.state.state, INPUT_DIMENSION)])

        preds = self.model.predict(input_to_model)
        value_array = preds[0]
        logits_array = preds[1]
        value = value_array[0]
        logits = logits_array[0]

        allowed_actions = leaf.allowed_actions

        if not allowed_actions.any():
            import ipdb; ipdb.set_trace()

        logits[np.logical_not(allowed_actions)] = -100

        # SOFTMAX
        odds = np.exp(logits)

        # print(allowed_actions)
        # print(logits)

        try:
            1 / np.sum(odds)
        except:
            import ipdb; ipdb.set_trace()

        probs = odds / np.sum(odds)

        return 244 * value, probs

    def evaluate_leaf(self, leaf):
        if leaf.done:
            return

        value, probs = self.get_preds(leaf)

        # print(value)  # , probs)

        for idx, allowed in enumerate(leaf.allowed_actions):
            if allowed:
                new_value, new_done, new_state = take_action(leaf.state, CARDS[idx])
                if new_done:
                    leaf.done = new_done
                    leaf.value = new_value
                    leaf.state = new_state
                    return
                if new_state.id not in self.mcts.tree:
                    node = mc.Node(new_state, value[0], new_done)
                    self.mcts.add_node(node)
                else:
                    node = self.mcts.tree[new_state.id]

                new_edge = mc.Edge(leaf, node, probs[idx], idx)
                leaf.edges.append(new_edge)

    def get_av(self, tau):
        edges = self.mcts.root.edges
        pi = np.zeros(self.action_size, dtype=np.integer)
        values = np.zeros(self.action_size, dtype=np.float32)
        for edge in edges:
            pi[edge.action] = pow(edge.stats['N'], 1.0 / tau)
            values[edge.action] = edge.stats['Q']

        pi = pi / np.sum(pi)
        return pi, values

    def choose_action(self, pi, values, tau):
        if tau == 0:
            actions = np.argwhere(pi == max(pi))
            action = random.choice(actions)[0]
        else:
            action_idx = np.random.multinomial(1, pi)
            action = np.where(action_idx == 1)[0][0]

        value = values[action]

        return action, value

    def replay(self, ltmemory, job):
        minibatch = random.sample(ltmemory, min(job['BATCH_SIZE'], len(ltmemory)))

        training_states = np.array([np.reshape(row['state'].state, INPUT_DIMENSION) for row in minibatch])
        training_targets = {
            'value_head': np.array([row['value'] / 244.0 for row in minibatch]),
            'policy_head': np.array([row['action_values'] for row in minibatch])
        }

        # print(training_targets['value_head'])
        # print(training_targets['policy_head'])

        # print(np.isfinite(training_states).all())

        if not np.isfinite(training_targets['value_head']).all():
            import ipdb; ipdb.set_trace()
        if not np.isfinite(training_targets['policy_head']).all():
            import ipdb; ipdb.set_trace()

        self.model.fit(
            training_states,
            training_targets,
            epochs=job['EPOCHS'],
            verbose=1,
            validation_split=0,
            batch_size=32
        )

    def predict(self, inputToModel):
        preds = self.model.predict(inputToModel)
        return preds
