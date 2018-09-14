import numpy as np

from code.model.residual_nn import Residual_CNN
import code.mcts.mcts as mc
from code.constants import sort_cards, encode_binary
from code.constants import CARDS
from code.rules import possible_cards
from code.gamm.game import evaluate_trick
from code.players.random_player import RandomPlayer


class ModelPlayer:
    def __init__(self, action_size, cpuct, mcts_simulations, config, model_file=None):
        self.action_size = action_size
        self.cpuct = cpuct
        self.MCTSsimulations = mcts_simulations
        self.config = config
        self.model = self.create_model(model_file)
        self.mcts = None

    def create_model(self, model_file):
        if model_file is not None:
            # Load from file
            print("Not implemented")
            return
        return Residual_CNN(self.config)

    def opponent_choose_card(self, cards, current_trick, game_type):
        return RandomPlayer().choose_card(cards, current_trick, game_type)

    def take_action(self, card, leaf):
        leaf.current_trick.append(card)
        leaf.track.append(card)
        leaf.current_position + 1

        opp_one_card = self.opponent_choose_card(leaf.opp_one_cards, leaf.current_trick, leaf.game_type)
        opp_two_card = self.opponent_choose_card(leaf.opp_two_cards, leaf.current_trick, leaf.game_type)

        if len(leaf.current_trick) == 3:
            current_position = evaluate_trick(leaf.winner, leaf.current_trick, leaf.game_type)
            current_trick = []

        state, state_id, allowed_actions = self.transform_to_state(
            leaf.game_type, current_position, leaf.player_pos,
        )

    def allowed_actions(self, cards, current_trick, game_type):
        allowed_cards = possible_cards(cards, current_trick, game_type)
        allowed_actions = np.zeros(32, dtype=bool)
        for c in allowed_cards:
            allowed_actions[CARDS.index(c)] = True
        return allowed_actions

    def transform_to_state(self, game_type, current_position, player_pos, cards, current_trick, track):
        state = list()
        for i in range(3):
            if i == player_pos:
                state.append([1] * 32)
            else:
                state.append([0] * 32)

        for i in range(3):
            if i == current_position:
                state.append([1] * 32)
            else:
                state.append([0] * 32)

        for card in sort_cards(cards):
            state.append(encode_binary([card]))
        for card in range(10 - len(cards)):
            state.append([0] * 32)

        for card in track:
            state.append(encode_binary([card]))
        for card in range(10 - len(track)):
            state.append([0] * 32)

        for card in current_trick:
            state.append(encode_binary([card]))
        for card in range(3 - len(current_trick)):
            state.append([0] * 32)

        state_id = ('').join([str(i) for sublist in state for i in sublist])
        allowed_actions = self.allowed_actions(cards, current_trick, game_type)

        return state, state_id, allowed_actions

    def build_mcts(self, state_id, state, allowed_actions, winner, game_type,
                   current_position, player_pos, cards, current_trick, track):
        self.root = mc.Node(
            state_id, state, allowed_actions, winner, game_type,
            current_position, player_pos, cards, current_trick, track
        )
        self.mcts = mc.MCTS(self.root, self.cpuct)

    def change_root_mcts(self, state_id):
        self.mcts.root = self.mcts.tree[state_id]

    def simulate(self):
        # MOVE THE LEAF NODE
        leaf, value, done, breadcrumbs = self.mcts.move_to_leaf()

        # EVALUATE THE LEAF NODE
        value, breadcrumbs = self.evaluate_leaf(leaf, value, done, breadcrumbs)

        # BACKFILL THE VALUE THROUGH THE TREE
        self.mcts.backFill(leaf, value, breadcrumbs)

    def choose_card(self, winner, game_type, current_position, player_pos, cards,
                    opp_one_cards, opp_two_cards, current_trick, track, tau=1):
        state, state_id, allowed_actions = self.transform_to_state(
            game_type, current_position, player_pos, cards, current_trick, track
        )
        if self.mcts is None or state_id not in self.mcts.tree:
            self.build_mcts(
                state_id, state, allowed_actions, winner, game_type,
                current_position, player_pos, cards, current_trick, track,
                opp_one_cards, opp_two_cards
            )
        else:
            self.change_root_mcts(state_id)

        for sim in range(self.MCTSsimulations):
            self.simulate()

        # get action values
        pi, values = self.getAV(1)

        # pick the action
        action, value = self.chooseAction(pi, values, tau)

        nextState, _, _ = state.takeAction(action)

        nn_value = -self.get_preds(nextState)[0]

        return (action, pi, value, nn_value)

    def get_preds(self, leaf):
        # predict the leaf
        input_to_model = np.array([np.reshape(leaf.state, (32, 29, 1))])

        preds = self.model.predict(input_to_model)
        value_array = preds[0]
        logits_array = preds[1]
        value = value_array[0]
        logits = logits_array[0]

        allowed_actions = leaf.allowed_actions
        logits[allowed_actions] = -100

        # SOFTMAX
        odds = np.exp(logits)
        probs = odds / np.sum(odds)

        return value, probs

    def evaluate_leaf(self, leaf, value, done, breadcrumbs):
        if done == 0:
            value, probs = self.get_preds(leaf)

            probs = probs[leaf.allowed_actions]

            for idx, allowed in enumerate(leaf.allowed_actions):
                if allowed is True:
                    new_state, _, _ = self.take_action(CARDS[idx], leaf)
                    if new_state.id not in self.mcts.tree:
                        node = mc.Node(new_state)
                        self.mcts.addNode(node)
                    else:
                        node = self.mcts.tree[new_state.id]

                    newEdge = mc.Edge(leaf, node, probs[idx], action)
                    leaf.edges.append((action, newEdge))

        return (value, breadcrumbs)

    def getAV(self, tau):
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
