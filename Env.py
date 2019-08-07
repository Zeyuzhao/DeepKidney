# %%

import random

import numpy as np

from math import sqrt, log

from GCN import *


# %%

### Edge List Implementation

class MaxSetState:
    def __init__(self, data, state=None):
        self.data = data
        self.node_weights = data["x"]
        self.edge_index = data["edge_index"].transpose()

        # number of total nodes in original graph
        self.num_nodes = len(data["x"])
        if state is None:

            self.state = np.ones(self.num_nodes, dtype=bool)
        else:
            self.state = state

    def clone(self):
        return MaxSetState(self.data, self.state.copy())

    def step(self, action):
        assert self.state[action] == 1

        edges = self.edge_index.transpose()
        # Get all the neighbors of node action
        removal = edges[edges[:, 0] == action][:, 1]

        # Create mask to remove node action and its neighbors
        mask = np.zeros(self.num_nodes, dtype=bool)
        mask[removal] = 1
        mask[action] = 1

        # Perform mask
        new_state = self.state & (~mask)
        reward = self.node_weights[action]
        return MaxSetState(self.data, new_state), reward

    def getNeighbors(self, node):
        edges = self.edge_index
        # Get all the neighbors of node action
        neighbors = edges[edges[:, 0] == node][:, 1]
        return neighbors

    def actions(self):
        return list(np.flatnonzero(self.state))

    def sample_action(self):
        return random.choice(self.actions())

    def get_edgelist(self):
        new_edges = []
        for e in self.edge_index:
            if self.state[e[0]] == 1 and self.state[e[1]] == 1:
                new_edges.append(e)
                # print("appended: " + str(e))
        return np.array(new_edges).transpose()

    def score(self):
        assert not self.state.any()
        return 0

    def __repr__(self):
        return str(self.state)


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

