# %%

from Env_adj import *
from GCN import *
from torch_geometric import data
visualize = MaxIndDataset('data/weighted_4')


for ID in range(100):

    test_item = visualize[ID]
    processed_test_item = {k: np.round(v.cpu().data.numpy(), 3) for k, v in iter(test_item)}
    processed_test_item["adj_matrix"] = edgeToAdj(len(processed_test_item["x"]), processed_test_item["edge_index"])
    state = MaxSetState(processed_test_item)

    state_item = {}
    state_item["x"] = test_item["x"]

    target = sum(processed_test_item['x'][processed_test_item['y'].astype(bool)])
    total_reward = 0

    correct = np.flatnonzero(processed_test_item['y'])

    nodes_chosen = 0
    while len(state.actions()) > 0:

        edge_index, node_weights = state.getEdgeIndex()
        node_weights = torch.tensor(node_weights).to(dtype=torch.float)
        edge_index = torch.tensor(edge_index).to(dtype=torch.long)
        state_item = data.Data(node_weights, edge_index)
        # print("Actions available: " + str(state.actions()))
        # print("Current Edge Index: " + str(state_item['edge_index']))

        # draw_entry(state_item, node_dict=state.actions())
        # action = -1
        # while not action in state.actions():
        #     action = int(input("Query Action: "))

        action = correct[nodes_chosen]
        # print("Action: "+ str(action))
        nodes_chosen += 1
        # print("Ranking: " + str(ranking))

        state, reward = state.step(action)
        total_reward += reward
        # print("New State: " + str(state))
        # print("Reward: " + str(reward))


    print("Target: {0:.3f} Total: {1:.3f} ".format(target, total_reward))
    if target != total_reward:
        print("ERROR!!!")

