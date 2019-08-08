# %%

import random

import numpy as np

from math import sqrt, log



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

    def step(self, action):
        assert self.state[action] == 1
        mask = self._adj_matrix[action].astype(bool)
        mask[action] = 1
        next_state = self.state & ~mask
        return MaxSetState(self.data, next_state), self.node_weights[action]

    # Returns the nodes in the graph
    def actions(self):
        return np.flatnonzero(self.state)

    def sample_action(self):
        return random.choice(self.actions())

    @property
    def _adj_matrix(self):
        temp = self.adj_matrix.copy()
        temp[~self.state] = 0
        temp[:, ~self.state] = 0
        return temp

    def getAdjMatrix(self):
        # Takes the columns and the rows remaining in the bitmask state variable
        return self.adj_matrix[self.state == 1][:, self.state == 1]

    def score(self):
        assert not self.state.any()
        return 0

    def __repr__(self):
        return str(self.state)

    def getShiftedIndex(self, index):
        return self.actions()[index]

    def getEdgeIndex(self):
        edge_list = []
        adj_matrix = self.getAdjMatrix()
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix)):
                if adj_matrix[i, j] == 1:
                    edge_list.append([i, j])
        return np.array(edge_list).transpose().reshape((2, -1)), self.node_weights[self.state]



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
        self.untriedActions = list(state.actions())

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

    ID = 5
    test_graph = visualize[ID]

    draw_entry(test_graph)

    test_data = {k: np.round(v.data.numpy(), 3) for k, v in iter(test_graph)}
    target = sum(test_data['x'][test_data['y'].astype(bool)])
    adj = edgeToAdj(len(test_data['x']), test_data["edge_index"])

    input_data = test_data.copy()
    input_data["adj_matrix"] = adj

    current = MaxSetState(data=input_data)

    edges = adjToEdge(current.getAdjMatrix())
    current, reward = current.step(0)
    current_adj = current.getAdjMatrix()
    num_nodes = len(current_adj)

    net_data = {}
    net_data['x'] = test_graph['x']
    net_data['edge_index'] = torch.tensor(adjToEdge(current_adj))







