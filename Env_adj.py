# %%

import random

import numpy as np

from math import sqrt, log

from GCN import *


# %%

### Adj Matrix List Implementation

class MaxSetState:
    def __init__(self, data, state = None):

        self.data = data
        self.adj_matrix = data["adj_matrix"]
        self.node_weights = data["x"]

        if state is None:
            self.state = np.ones(len(data["adj_matrix"]), dtype=bool)
        else:
            self.state = state

    def clone(self):
        return MaxSetState(self.adj_matrix, self.state.copy())

    # action is the node being removed

    def shiftedIndex(self, mask, indices):
        removed = np.flatnonzero(mask)
        new_index = [0] * len(indices)

        # Edge Cases
        if len(removed) == 0:
            return indices

        if len(removed) == 1:
            for i in range(len(removed)):
                if indices[i] >= removed[0]:
                    indices[i] += 1
            return indices

        spacings = np.diff(removed) - 1
        s = np.cumsum(spacings)

        for i in range(len(indices)):
            if indices[i] < removed[0] - 1:
                new_index[i] = indices[i]
            else:
                new_index[i] = indices[i] + np.searchsorted(s, indices[i])
                print("")

        return new_index

    def step(self, action):
        assert self.state[action] == 1
        temp_mask = self.getAdjMatrix()[action].astype(bool)
        indices = self.shiftedIndex(~self.state, np.flatnonzero(temp_mask))

        mask = np.zeros(len(self.state)).astype(bool)
        mask[indices] = 1
        mask[action] = 1
        new_state = self.state & (~mask) # Remove the action node and its neighbors
        print("State before freebies: " + str(new_state))

        # Check for any isolated free nodes
        current_adj = self.stateToAdjMatrix(new_state)
        print("Adj Matrix: \n" + str(current_adj))

        freebies = np.where(~current_adj.any(axis=1))[0]

        true_freebies = self.shiftedIndex(~new_state, freebies)
        print("Shifted Freebies: " + str(true_freebies))
        # Remove the free nodes
        new_state[true_freebies] = 0

        reward = self.node_weights[action] + sum([self.node_weights[f] for f in freebies])
        return MaxSetState(self.data, new_state), self.node_weights[action]

    # Returns the nodes in the graph
    def actions(self):
        return list(np.flatnonzero(self.state))

    def sample_action(self):
        return random.choice(self.actions())

    def getAdjMatrix(self):
        # Takes the columns and the rows remaining in the bitmask state variable
        return self.stateToAdjMatrix(self.state)

    def stateToAdjMatrix(self, updatedState):
        # Takes the columns and the rows remaining in the bitmask state variable
        return self.adj_matrix[updatedState == 1][:, updatedState == 1]

    def score(self):
        assert not self.state.any()
        return 0
    def __repr__(self):
        return str(self.state)


def edgeToAdj(num_nodes, edge_index):
    adj_mat = np.zeros((num_nodes, num_nodes))
    for edge in edge_index.transpose():
        adj_mat[edge[0], edge[1]] = 1
    return adj_mat

def adjToEdge(adj_matrix):
    edge_list = []
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if adj_matrix[i, j] == 1:
                edge_list.append([i, j])
    return np.array(edge_list).transpose()

# %%

class Node:
    def __init__(self, state: MaxSetState, action=None, parent=None):
        # The action that led to this state, and a reference to the parent
        self.action = action
        self.parentNode = parent

        # Statistics
        self.totalScore = 0
        self.visits = 1

        # Children of node
        self.childNodes = []
        self.untriedActions = state.actions()

    def select_child(self):
        upper_confidence = {c: c.totalScore / c.visits + sqrt(4 * log(self.visits) / c.visits) for c in self.childNodes}
        return max(upper_confidence, key=upper_confidence.get)

    def addChild(self, state, action):
        n = Node(state, action, self)
        self.untriedActions.remove(action)
        self.childNodes.append(n)
        return n

    def update(self, score):
        self.visits += 1
        self.totalScore += score

    def __repr__(self):
        return "[A: {0} S/V: {1:.2f}/{2} AR: {4:.2f} U: {3}]".format(self.action, self.totalScore, self.visits - 1,
                                                                     self.untriedActions, self.totalScore / self.visits)

    def treeToString(self, level=0):
        s = '\t' * level + "" + str(self) + "\n"
        for c in self.childNodes:
            s += c.treeToString(level + 1)
        return s


# %%

def search(root: MaxSetState, itermax, verbose):
    rootnode = Node(root)

    for i in range(itermax):
        if (verbose == 2): print(rootnode.treeToString(0))
        node = rootnode
        state = root.clone()
        # Selection
        while node.untriedActions == [] and node.childNodes != []:
            node = node.select_child()
            state, reward = state.step(node.action)

        # Expansion
        if node.untriedActions != []:
            a = random.choice(node.untriedActions)
            if (verbose == 2):
                print("Taking action: " + str(a))
            state, reward = state.step(a)
            node = node.addChild(state, a)

        # Simulation
        if (verbose == 2):
            print("Starting State: " + str(state))
        score = 0
        while state.actions() != []:
            sampled_action = state.sample_action()
            state, reward = state.step(sampled_action)
            score += reward
            if (verbose == 2):
                print("Simulating Action: " + str(sampled_action))
                print("Simulating Reward: " + str(reward))
                print("Next State: " + str(state))
        # Backpropagate

        while node.action != None:
            score += state.node_weights[node.action]
            node.update(score)
            if (verbose == 2):
                print("Node Weights: " + str(state.node_weights))
                print("Action: " + str(node.action))
                print("Reward Value: " + str(state.node_weights[node.action]))
            node = node.parentNode

    if (verbose and verbose != 2): print(rootnode.treeToString(0))

    actions = {c.action: c.visits for c in rootnode.childNodes}
    return sorted(list(actions.keys()), key=actions.get, reverse=True)

if __name__ == '__main__':
    from GCN import *

    visualize = MaxIndDataset('data/weighted_4')

    ID = 4
    test_graph = visualize[ID]

    # draw_entry(test_graph)

    test_data = {k: np.round(v.data.numpy(), 3) for k, v in iter(test_graph)}
    target = sum(test_data['x'][test_data['y'].astype(bool)])
    adj = edgeToAdj(len(test_data['x']), test_data["edge_index"])

    input_data = test_data.copy()
    input_data["adj_matrix"] = adj

    current = MaxSetState(data=input_data)

    print("Tested: " + str(current.shiftedIndex([1, 0, 1], [0])))
    next = current.step(0)
    print(next[0].getAdjMatrix())


