import random

from math import sqrt, log

from v1_src.GCN_old import *

# In[1]
class MaxSetState:
    def __init__(self, adj_matrix, state = None):
        if state is None:
            self.state = np.ones(len(adj_matrix), dtype=bool)
        else:
            self.state = state
        self.adj_matrix = adj_matrix

    def clone(self):
        return MaxSetState(self.adj_matrix, self.state.copy())

    def step(self, action):
        assert self.state[action] == 1
        mask = self.adj_matrix[action].astype(bool)
        mask[action] = 1

        newState = self.state & (~mask) # Remove the acting node and its neighbors

        #print(newState)
        return MaxSetState(self.adj_matrix, newState)

    def actions(self):
        return list(np.flatnonzero(self.state))

    def sample_action(self):
        return random.choice(self.actions())

    def currentAdjMatrix(self):
        # Takes the columns and the rows remaining in the bitmask state variable
        return self.adj_matrix[self.state == 1][:, self.state == 1]

    def score(self):
        assert not self.state.any()
        return 0
    def __repr__(self):
        return str(self.state)



class Node:
    def __init__(self, state: MaxSetState, action = None, parent = None):
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
        upper_confidence = {c: c.totalScore/c.visits + sqrt(4 * log(self.visits)/c.visits) for c in self.childNodes}
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
        return "[A: {0} S/V: {1}/{2} R: {4:.2f} U: {3}]".format(self.action, self.totalScore, self.visits - 1, self.untriedActions, self.totalScore / self.visits)

    def treeToString(self, level = 0):
        s = '\t' * level + "" + str(self) + "\n"
        for c in self.childNodes:
            s += c.treeToString(level + 1)
        return s


def search(root: MaxSetState, itermax, verbose):

    rootnode = Node(root)

    for i in range(itermax):
        node = rootnode
        state = root.clone()
        # Selection
        while node.untriedActions == [] and node.childNodes != []:
            node = node.select_child()
            state = state.step(node.action)

        # Expansion
        if node.untriedActions != []:
            a = random.choice(node.untriedActions)
            #print("action: " + str(a))
            state = state.step(a)
            node.addChild(state, a)

        # Simulation

        score = 0
        while state.actions() != []:
            state = state.step(state.sample_action())
            score += 1
        # Backpropagate

        while node != None:
            node.update(score)
            node = node.parentNode
            score += 1
    if (verbose): print(rootnode.treeToString(0))

    actions = {c.action: c.visits for c in rootnode.childNodes}
    return sorted(list(actions.keys()), key=actions.get, reverse=True)
# In[2]:

visualize = MaxIndDataset('data/small')

# big_set = MaxIndDataset('DeepKidney/data/binomial_80')

ID = 8
test_graph = visualize[ID]

target = sum(test_graph.y.numpy())
def edgeToMatrix(nodes, edge_index):
    mat = np.zeros((nodes, nodes))
    e = edge_index.numpy().T
    for item in e:
        mat[item[0], item[1]] = 1
    return mat

test_mat = edgeToMatrix(len(test_graph.x), test_graph.edge_index)
# In[3]:

tot_score = 0
trials = 20
for i in range(trials):
    start = MaxSetState(adj_matrix=test_mat)
    score = 0
    action = "None"
    while start.actions() != []:
        result = search(start, 5000, 1)
        start = start.step(result[0])
        print("Action: {0} Results: {1}".format(action, result))
        action = result[0]
        score += 1

    tot_score += score
    print("Action: {0} Score: {1}".format(action, score))
    print("Target: " + str(target))
    print("--------------------")
print("Algo Trial Score: {0}".format(tot_score / trials))



# In[4]:
#
# for i in range(20):
#     draw_entry(visualize[i], title=str(i))
#draw_entry(visualize[ID], title=str(ID))

# Control: Random Selection


def random_rollout(s):
    score = 0
    state = s.clone()
    while state.actions() != []:
        state = state.step(state.sample_action())
        score += 1
    return score

start = MaxSetState(adj_matrix=test_mat)

rand_trials = 1000
rand_tot = 0
for i in range(rand_trials):
    rand_tot += random_rollout(start)

print("Random Trial Score: {0}".format(rand_tot/rand_trials))