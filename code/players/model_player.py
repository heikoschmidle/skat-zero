import json
import numpy as np

import configuration

from model.residual_nn import Residual_CNN
import model.mcts

class ModelPlayer:
    def __init__(self, action_size, cpuct, mcts_simulations, model_file=None):
        self.action_size = action_size
        self.cpuct = cpuct
        self.MCTSsimulations = mcts_simulations
        self.model = self.create_model(model_file)
        self.mcts = None
        self.input_state = self.create_input_state()

    def create_model(self, model_file):
        if model_file is not None:
            # Load it from file
            print("Not implemented")
            return
        return Residual_CNN()

    def create_input_state(self):
        return np.reshape(np.zeros(32 * 3 * 11), (32, 3, 11))

    def transform_to_state(self, current_position, player_pos, cards, current_trick, track):
        state = list()
        for i in range(3):
            if i == player_pos:
                state.append([1] * 32)
            else:
                state.append([0] * 32)
        import ipdb; ipdb.set_trace()
        for card in track:
            print(card)

    def build_mcts(self, state):
        self.root = mc.Node(state)
        self.mcts = mc.MCTS(self.root, self.cpuct)

    def change_root_mcts(self, state):
        self.mcts.root = self.mcts.tree[state.id]

    def simulate(self):
        # MOVE THE LEAF NODE
        leaf, value, done, breadcrumbs = self.mcts.moveToLeaf()

        # EVALUATE THE LEAF NODE
        value, breadcrumbs = self.evaluateLeaf(leaf, value, done, breadcrumbs)

        # BACKFILL THE VALUE THROUGH THE TREE
        self.mcts.backFill(leaf, value, breadcrumbs)

    def choose_card(self, current_position, player_pos, cards, current_trick, track, tau=1):
        state = self.transform_to_state(current_position, player_pos, cards, current_trick, track)
        if self.mcts is None or state.id not in self.mcts.tree:
            self.build_mcts(state)
        else:
            self.change_root_mcts(state)

        # run the simulation
        for sim in range(self.MCTSsimulations):
            self.simulate()

        # get action values
        pi, values = self.getAV(1)

        # pick the action
        action, value = self.chooseAction(pi, values, tau)

        nextState, _, _ = state.takeAction(action)

        NN_value = -self.get_preds(nextState)[0]

        return (action, pi, value, NN_value)

    def get_preds(self, state):
        # predict the leaf
        inputToModel = np.array([self.model.convertToModelInput(state)])

        preds = self.model.predict(inputToModel)
        value_array = preds[0]
        logits_array = preds[1]
        value = value_array[0]
        logits = logits_array[0]

        allowedActions = state.allowedActions

        mask = np.ones(logits.shape, dtype=bool)
        mask[allowedActions] = False
        logits[mask] = -100

        # SOFTMAX
        odds = np.exp(logits)
        probs = odds / np.sum(odds)  # put this just before the for?

        return ((value, probs, allowedActions))

    def evaluateLeaf(self, leaf, value, done, breadcrumbs):
        if done == 0:
            value, probs, allowedActions = self.get_preds(leaf.state)

            probs = probs[allowedActions]

            for idx, action in enumerate(allowedActions):
                newState, _, _ = leaf.state.takeAction(action)
                if newState.id not in self.mcts.tree:
                    node = mc.Node(newState)
                    self.mcts.addNode(node)
                else:
                    node = self.mcts.tree[newState.id]

                newEdge = mc.Edge(leaf, node, probs[idx], action)
                leaf.edges.append((action, newEdge))
        return ((value, breadcrumbs))

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
