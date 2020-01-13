from model.EvalUtils import max_eval
from model.GCN import ConvNet3
from search.MCSearch import *
from torch_geometric import data


def random_rollout(data):
    score = 0
    state = MaxSetState.loadFromTensor(data)
    while len(state.actions()) > 0:
        state, reward = state.step(state.rand_action())
        score += reward
    return score

def random_search(data, iter):
    score = 0
    state = MaxSetState.loadFromTensor(data)
    while len(state.actions()) > 0:
        a, _ = search(state, None, iter, 1, random_rollout=True)
        state, reward = state.step(a[0])
        score += reward
    return score

def greedy_rollout(data):
    score = 0
    state = MaxSetState.loadFromTensor(data)
    while len(state.actions()) > 0:
        a = greedy_action(state)
        # Convert the indices from the subgraph back to the original graph
        action = state.actions()[a]
        state, reward = state.step(action)
        #print("Greedy Action: {0} Reward: {1}".format(action, reward))
        score += reward
    return score

def greedy_action(state: MaxSetState):
    t_item = state.getTensorItem()
    g = to_networkx(t_item)
    loss = 99999999
    index = -1
    weights = t_item["x"].cpu().data.numpy()
    for n in g.nodes:
        l = sum([weights[o] for o in g.neighbors(n)]) - weights[n]
        if l < loss:
            index = n
            loss = l
    return index


def random_rollout_avg(data, rand_trials):
    tot_rand_score = 0
    for i in range(rand_trials):
        tot_rand_score += random_rollout(data)
    rand_score = tot_rand_score / rand_trials
    return rand_score


def random_rollout_max(data, rand_trials):
    max_rand_score = 0
    for i in range(rand_trials):
        max_rand_score = max(max_rand_score, random_rollout(data))
    return max_rand_score



def searchTester(dataset_name):
    visualize = KidneyDataset(dataset_name)
    pi = MaximalNetWrapper()
    pi.load_checkpoint(name="run_1/size_mixed/epoch_9.pt")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conv_net = ConvNet3().to(device)
    model_folder = "../../model/run_1/"
    model_name = "size_mixed"
    model_file = "epoch_9.pt"
    params = torch.load(osp.join(model_folder, model_name, model_file))
    conv_net.load_state_dict(params)
    conv_net.eval()

    for ID in range(0, 100):
        test_graph = visualize[ID]
        #draw_MIS(test_graph, title="ID: {0}".format(ID))
        tot_score = 0
        max_tot_score = 0
        trials = 1

        test_data = {k: np.round(v.data.numpy(), 3) for k, v in iter(test_graph)}

        test_data["adj_matrix"] = edgeToAdj(len(test_data["x"]), test_data["edge_index"])
        target = sum(test_data['x'][test_data['y'].astype(bool)])

        for i in range(trials):
            state = visualize.state_repr(ID)
            score = 0
            action = "None"
            reward = 0
            chosen = []
            while len(state.actions()) > 0:
                tensor_item = state.getTensorItem()
                disp_item = data.Data(tensor_item["x"], tensor_item["edge_index"]).to("cpu")
                #draw_entry(disp_item, title="ID: {0} Size: {1} Action: {2}".format(ID, len(tensor_item["x"]), action))
                result, A_DEBUG = search(state, pi, 200, 1, random_rollout=True)
                state, r = state.step(result[0])
                if TEST:
                    print("Action: {0} Reward: {1:.3f} Length: {2}".format(action, reward, len(state.actions())))
                    print(r)
                    #input()
                action = result[0]
                score += r
                reward = r
                chosen.append(action)
            tot_score += score

            optimal = np.flatnonzero(test_data['y'])
            if TEST:
                print("Chosen: {0}".format(sorted(chosen)))
                print("Optimal: {0}".format(sorted(optimal)))
                print("Action: {0} Score: {1:.3f}".format(action, score))
                print("Target: {0:.3f}".format(target))
                #rand_max_baseline = random_rollout_max(test_data, 1000)
                #print("Rand Max: {0:.3f}".format(rand_max_baseline))
                print("--------------------")
            max_tot_score = max(max_tot_score, score)
        greedy_baseline = greedy_rollout(test_graph)
        #print("Greedy: " + str(greedy_baseline))
        rand_avg_baseline = random_rollout_avg(test_graph, 500)
        rand_max_baseline = random_rollout_max(test_graph, 500)
        rand_search = None #random_search(test_graph, 10)
        model_baseline, actions = max_eval(test_graph, conv_net)
        input()
        print("Model Baseline: {0} Actions: {1}".format(model_baseline, actions))
        print("ID[{0}]: Target: {1:.3f} Algo Avg Score: {2:.3f}\t Algo Max Score: {3:.3f}\t Rand Avg Score: {4:.3f} Rand Max Score: {5:.3f} Rand Search: {6:.3f} Greedy Score {7}".format(ID, target, (tot_score / trials), max_tot_score, rand_avg_baseline, rand_max_baseline, rand_search, greedy_baseline))

TEST = True
if __name__ == '__main__':
    searchTester("../../data/dense_100/weighted_n100_p100")
