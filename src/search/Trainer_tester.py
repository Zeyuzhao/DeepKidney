from search.Trainer import *


def random_rollout(data):
    score = 0
    state = MaxSetState(data)
    while len(state.actions()) > 0:
        state, reward = state.step(state.sample_action())
        score += reward
    return score

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
    visualize = MaxIndDataset(dataset_name)
    pi = MaximalNetWrapper()
    pi.load_checkpoint(name="convnet3/weighted_convnet_epoch_99.pt")

    for ID in range(100):
        test_graph = visualize[ID]
        draw_entry(test_graph)
        tot_score = 0
        max_tot_score = 0
        trials = 5

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
                result, A_DEBUG = search(state, pi, 1000, 1)

                state, r = state.step(result[0])
                print("Action: {0} Reward: {1:.3f} Results: {2}".format(action, reward, result))
                print(r)
                action = result[0]
                score += r
                reward = r
                chosen.append(action)
            tot_score += score

            optimal = np.flatnonzero(test_data['y'])
            print("Chosen: {0}".format(sorted(chosen)))
            print("Optimal: {0}".format(sorted(optimal)))
            print("Action: {0} Score: {1:.3f}".format(action, score))
            print("Target: {0:.3f}".format(target))
            rand_max_baseline = random_rollout_max(test_data, 10000)
            print("Rand Max: {0:.3f}".format(rand_max_baseline))
            print("--------------------")
            max_tot_score = max(max_tot_score, score)
        rand_avg_baseline = random_rollout_avg(test_data, 1000)
        rand_max_baseline = random_rollout_max(test_data, 1000)
        print("ID[{0}]: Target: {1:.3f} Algo Avg Score: {2:.3f}\t Algo Max Score: {3:.3f}\t Rand Avg Score: {4:.3f} Rand Max Score: {5:.3f}".format(ID, target, (tot_score / trials), max_tot_score, rand_avg_baseline, rand_max_baseline))

if __name__ == '__main__':
    searchTester("../../data/weighted_120")
