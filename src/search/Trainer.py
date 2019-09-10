from env.Env import *
from model.MaximalNet import MaximalNetWrapper
from data.dataset import *
from tqdm import tqdm

import json


def softmax(array, temp):
    exp = np.exp(array / temp)
    return exp / sum(exp)

def search(root: MaxSetState, pi, itermax, verbose, random_rollout=False):
    rootnode = Node(root, pi=pi)

    for i in tqdm(range(itermax)):
        #if (verbose == 2): print(rootnode.treeToString(0))
        node = rootnode
        state = root.clone()
        # Selection
        if (verbose == 2):
            print("Selection vvvvvvvvv")
        while not node.untried_mask.any() and node.childNodes != []:
            node = node.select_child()
            if (verbose == 2):
                print("Selection Action: " + str(node.action))
            state, reward = state.step(node.action)

        # Expansion
        if (verbose == 2):
            print("Expansion vvvvvvvvv")

        if node.untried_mask.any():
            a = node.select_expansion_action()
            state, reward = state.step(a)
            node = node.addChild(state, a)
            if (verbose == 2):
                print("Expanding action: " + str(a))
                print("Node Priors : " + str(node._priors))

        # Simulation

        if (verbose == 2):
            print("Simulation vvvvvvvvv")
            print("Starting State: " + str(state))
        score = 0
        while len(state.actions()) > 0:
            if pi and not random_rollout:
                index, weights = state.getEdgeIndex()
                graph = {
                    "x": weights,
                    "edge_index": index
                }
                output = pi.predict(graph).flatten()

                probs = softmax(output, 3)

                if (verbose == 2):
                    print("Weights: {0}".format(weights))
                    print("Policy Probs: " + str(probs.transpose()))
                action = np.random.choice(state.actions(), p = probs)
            else:
                action = state.rand_action()
            state, reward = state.step(action)
            score += reward

            if (verbose == 2):
                print("Simulating Action: " + str(action))
                print("Simulating Reward: " + str(reward))
                print("Next State: " + str(state))
                print("##################################")
        # Backpropagate
        if (verbose == 2):
            print("Backpropagation vvvvvvvvv")
        while node.action != None:
            score += state.node_weights[node.action]
            node.update(score)
            if (verbose == 2):
                #print("Node Weights: " + str(state.node_weights))
                print("Action: " + str(node.action))
                print("Reward: " + str(state.node_weights[node.action]))
                print("Value: " + str(score))
                print("**************************")
            node = node.parentNode
        # Update the root node
        node.update(score)
        if (verbose == 2):
            tree = rootnode.getTree()
            #print(json.dumps(rootnode.getTree(), indent=2, sort_keys=False))
            print(rootnode.getChildInfo())
            input()
            pass



    #if (verbose): print(rootnode.treeToString(0))

    actions = {c.action: c.visits for c in rootnode.childNodes}
    return sorted(list(actions.keys()), key=actions.get, reverse=True), rootnode.getTree()
    #return actions

if __name__ == '__main__':
    visualize = MaxIndDataset('../../data/weighted_4')

    ID = 0
    state = visualize.state_repr(ID)
    soln = visualize[ID]["y"].cpu().data.numpy()
    soln_elements = np.flatnonzero(soln)
    target = sum(state.node_weights[soln.astype(bool)])

    draw_entry(visualize[ID])
    pi = MaximalNetWrapper()
    pi.load_checkpoint(name="convnet3/weighted_convnet_epoch_99.pt")

    results = search(state, pi, 100, 2)

