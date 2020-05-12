# %%

from collections import defaultdict
from math import sqrt
import torch

# %%

### Adj Matrix List Implementation
from torch_geometric.data import Data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MaxSetState:
    def __init__(self, data, state = None):

        self.data = data
        self.adj_matrix = data["adj_matrix"]
        self.node_weights = data["x"]

        if state is None:
            self.state = np.ones(len(data["adj_matrix"]), dtype=bool)
        else:
            self.state = state

    @staticmethod
    def loadFromTensor(tensor_item):
        data = {}
        edge_index = tensor_item["edge_index"].cpu().data.numpy()
        data["x"] = tensor_item["x"].cpu().data.numpy()
        data["adj_matrix"] = edgeToAdj(len(tensor_item["x"]), edge_index)
        return MaxSetState(data)

    def clone(self):
        return MaxSetState(self.data, self.state.copy())

    def step(self, action):
        assert self.state[action] == 1
        mask = self._adj_matrix[action].astype(bool)
        mask[action] = 1
        next_state = self.state & ~mask
        return MaxSetState(self.data, next_state), self.node_weights[action]

    # Returns the nodes in the graph
    def actions(self):
        return np.flatnonzero(self.state)

    def prob_action(self, prob_arr):
        prob_arr = prob_arr / sum(prob_arr)
        return np.random.choice(self.actions(), p = prob_arr)

    def rand_action(self):
        return np.random.choice(self.actions())

    def max_action(self, prob_arr):
        return self.actions()[np.argmax(prob_arr)]

    def getTensorItem(self):
        tensor_item = Data()
        edge_index, x = self.getEdgeIndex()
        tensor_item["edge_index"] = torch.tensor(edge_index).to(device, dtype=torch.long)
        tensor_item["x"] = torch.tensor(x).to(device, dtype=torch.float)
        tensor_item.num_nodes = len(x)
        return tensor_item

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
        adj_matrix = self.getAdjMatrix()
        edge_list = np.nonzero(adj_matrix)
        if len(edge_list) == 0:
            edge_list = np.array([[], []])
        return edge_list, self.node_weights[self.state]



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

def softmax(array, temp):
    exp = np.exp(array / temp)
    return exp / sum(exp)

# %%
# Exploration parameter, c_puct
EXPLORE = 50
DEBUG = False
TEMP = 0.05
INIT_COUNT = 0.1
class Node:
    def __init__(self, state: MaxSetState, action=None, parent=None, pi = None, expansion_limit = 50):
        # The action that led to this state, and a reference to the parent
        self.action = action
        self.parentNode = parent
        self.state = state
        # Statistics
        self.totalScore = 0
        # Begin with a positive initialization to prevent divide-by-0 error
        self.visits = INIT_COUNT

        # Children of node
        self.childNodes = []

        self.num_actions = len(state.actions())

        self.untried_mask = np.ones(self.num_actions, dtype=bool)

        self.actions = state.actions()
        # Compute Priors
        self.pi = pi
        # Maps actions to their index in probability vector
        self.index = {}

        self.max_score = 0

        self.expansion_limit = expansion_limit

        for i, a in enumerate(self.actions):
            self.index[a] = i

        if pi:
            index, weights = state.getEdgeIndex()
            graph = {
                "x": weights,
                "edge_index": index
            }
            output = pi.predict(graph).flatten()
            probs = softmax(output, TEMP)
            #print("INDEX: " + str(self.index))
            #print(" PROBS: " + str(probs))
            self.setPriors(probs)
        else:
            self._priors = np.ones(self.num_actions)

        self.update_ucb()

    def select_child(self):
        self.update_ucb()
        #print("Upper Confidences: " + str(upper_confidence))
        return max(self.ucb, key=self.ucb.get)

    # We limit the number of nodes to expand
    # Returns boolean once the expansion_limit is reached
    def fullyExpanded(self):
        return (self.num_actions - np.count_nonzero(self.untried_mask)) > self.expansion_limit

    def select_expansion_action(self):
        untried = self.actions[self.untried_mask]
        norm_probs = self._priors[self.untried_mask] / sum(self._priors[self.untried_mask])
        action = np.random.choice(untried, p=norm_probs)
        return action

    def update_ucb(self):
        self.ucb = defaultdict(int)
        self.u_val = defaultdict(int)
        self.q_val = defaultdict(int)
        for c in self.childNodes:
            self.ucb[c] = c.max_score + EXPLORE * self._priors[self.index[c.action]] * sqrt(self.visits) / c.visits
            self.u_val[c] = EXPLORE * self._priors[self.index[c.action]] * sqrt(self.visits) / c.visits
            self.q_val[c] = c.max_score

    def addChild(self, state, action):
        n = Node(state, action, self, self.pi, expansion_limit=self.expansion_limit)
        self.untried_mask[self.index[action]] = 0
        self.childNodes.append(n)
        return n


    def update(self, score):
        self.visits += 1
        self.totalScore += score
        self.max_score = max(self.max_score, score)
        self.update_ucb() # May lag performance

    def getUCBScore(self):
        p = self.parentNode
        if p:
            ucb = p.ucb[self]
            return ucb
        else:
            return -1

    def getQUScore(self):
        p = self.parentNode
        if p:
            q = p.q_val[self]
            u = p.u_val[self]
            return q, u
        else:
            return self.totalScore / self.visits, -1

    def getPriorScore(self):
        p = self.parentNode
        if p:
            pr = p._priors[p.index[self.action]]
        else:
            pr = -1
        return pr

    def __repr__(self):
        if DEBUG:
            q, u = self.getQUScore()
            return "A{0}: {1} P: {2:.3f} Q: {3:.3f} U: {4:.3f} V: {5}".format(self.action, self.state, self.getPriorScore(), q, u, self.visits)
        else:
            p = self.parentNode
            if p:
                ucb = p.ucb[self]
            else:
                ucb = -1
            q_val = self.totalScore / self.visits
            u_val = ucb - q_val
            #q, u = self.getQUScore()
            #print("UCB: {0} Q: {1} U: {2} Derived Q: {3}".format(ucb, q, u, self.totalScore / self.visits))
            # if not isclose(q_val, q):
            #     raise Exception("u values not equal {0} {1}".format(u, u_val))

            prior_dict = dict(zip(self.actions, self._priors))
            return "[A: {0} S/V: {1:.2f}/{2} Q: {3:.2f} U: {4:.2f} Q+U: {5:.2f} P: {6}]".format(self.action, self.totalScore, self.visits - INIT_COUNT, q_val, u_val, ucb, prior_dict)

    def treeToString(self, level=0):
        s = '\t' * level + "" + str(self) + "\n"

        children = {c: c.visits for c in self.childNodes}
        children = sorted(list(children.keys()), key=children.get, reverse=True)
        for c in children:
            s += c.treeToString(level + 1)
        return s

    def getTree(self):
        prior_dict = {a: self._priors[self.index[a]] for a in self.actions}
        info = {
            "state": str(self),
        }
        children = {c: c.visits for c in self.childNodes}
        children = sorted(list(children.keys()), key=children.get, reverse=True)
        info["children"] = [c.getTree() for c in children]
        return info

    def getChildInfo(self):
        children = {c: c.visits for c in self.childNodes}
        children = sorted(list(children.keys()), key=children.get, reverse=True)
        out = ""
        for c in children:
            out += str(c) + "\n"
        return out

    def setPriors(self, probs):
        if (len(probs) != self.num_actions):
            raise("Prior prob dim does not equal action dim")
        self._priors = probs

if __name__ == '__main__':
    from v1_src.GCN_old import *

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







