from model.GCN import *
from data.dataset import *
from collections import OrderedDict
from tqdm import tqdm
from glob import glob
from model.EvalUtils import *
from search.MCSearch_tester import random_rollout_max, random_rollout_avg, greedy_rollout
import seaborn as sns

DEBUG = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

conv_net = ConvNet3().to(device)



model_folder= "../../model/run_1/"
model_name= "size_mixed"
model_file = "epoch_9.pt"

params = torch.load(osp.join(model_folder, model_name, model_file))
conv_net.load_state_dict(params)
conv_net.eval()

dataset_name = "kidney_fig_sparsity"
# dataset_name = "weighted_501"
dataset_folder = osp.join("../../data/", dataset_name)

dataset_subfolders = glob(osp.join(dataset_folder, "[!processed]*/"))

print(dataset_subfolders)

# Sorts the folder by getting the graph's sizes

# Format of each subfolder: [name]_10/
dataset_subfolders.sort(key=lambda x: int(x.split("_")[-1][:-1][1:]) if "_" in x else 0)

if len(dataset_subfolders) == 0:
    dataset_subfolders.append(dataset_folder)

# acc_arr = [0] * len(dataset_subfolders)
# opt_arr = [0] * len(dataset_subfolders)
#
# rand_arr = [0] * len(dataset_subfolders)
# compare_opt = [0] * len(dataset_subfolders)
#
# greedy_arr = [0] * len(dataset_subfolders)
# dataset_names = []

stats = pd.DataFrame(columns=["Method", "name", "score"])
for dataset_num, dataset_dir in enumerate(dataset_subfolders):
    test_set = KidneyDataset(dataset_dir)
    dataset_name = "0.0" + dataset_dir.split("/")[-2].split("_")[-1][1:]
    #dataset_name = dataset_dir.split("/")[-2].split("_")[-2][1:]

    num_evaled = 0
    for ID in tqdm(range(len(test_set))):
        test_item = test_set[ID]

        if (len(test_item["x"]) == 0):
            continue
        num_evaled += 1
        test_item.to(device)

        # Compute acc by comparing label to model's output
        out = conv_net(test_item)
        vals = out.detach().cpu().numpy().T.flatten()
        labels = test_item["y"].detach().cpu().numpy().T.flatten()
        #print(vals)
        bool_vals = vals >= 0.5
        correct = np.equal(bool_vals, labels)


        acc = np.count_nonzero(correct) / len(correct)

        # Compute a score for each test item by running through the MDP guided by network
        #score = 0
        # for i in range(5):
        #     s, actions = max_eval(test_item, conv_net)
        #     score = max(s, score)

        score, _ = max_eval(test_item, conv_net)
        greedy_score = greedy_rollout(test_item)
        rand_score = random_rollout_max(test_item, 1)
        # Get the sum of weights dictated by weighting
        label_score = sum(test_item["x"][test_item["y"].to(torch.bool)]).item()

        opt_ratios = {}
        opt_ratios["MCTS"] = score / label_score
        opt_ratios["Random"] = rand_score / label_score
        opt_ratios["Greedy"] = greedy_score / label_score
        # if rand_score == label_score:
        #     if score == label_score:
        #         opt_compare = 1
        #     else:
        #         opt_compare = 0
        # else:
        #     opt_compare = (score - rand_score) / (label_score - rand_score)
        # acc_arr[dataset_num] += acc
        # opt_arr[dataset_num] += score_ratio
        #
        # rand_arr[dataset_num] += rand_score / label_score
        # compare_opt[dataset_num] += opt_compare
        # greedy_arr[dataset_num] += greedy_score / label_score
        metrics = ["MCTS", "Greedy", "Random"]
        temp = []
        for m in metrics:
            stats = stats.append({
                "Method": m,
                "name": dataset_name,
                "score": opt_ratios[m]
            }, ignore_index=True)

        # metrics = ["GCN", "Greedy", "Random"]
        # temp = []
        # for m in metrics:
        #     temp.append([m, test_item["num_comp_nodes"], dataset_name, opt_ratios[m]])
        # entry = pd.DataFrame(temp, columns=stats.columns)
        # stats.append(entry, ignore_index=True)

        if DEBUG and acc < 0.99:
            # Sorts the indices from greatest value to the least
            rankings = np.flip(vals.argsort())
            dictionary = OrderedDict()
            #draw_entry(test_item.to(torch.device("cpu")), title="weighted_10: " + str(ID), node_color=correct, edge_color=labels)
            print("ID: " + str(ID))
            #print("Model Rankings: {0}".format(rankings))
            # for i in range(len(rankings)):
            #     dictionary[rankings[i]] = vals[rankings[i]]
            #print(dictionary)
            #print(vals)
            #print(correct)
            #print("Weights: " + str(test_item["x"].cpu().data.numpy()))
            print("Num Nodes: " + str(len(out)))
            print("GNN: " + str(score))
            print("Optimal: " + str(label_score))
            print("Random: " + str(rand_score))
            print("Greedy: " + str(greedy_score))
            print("**************************************")
            #input()
    # acc_arr[dataset_num] /= num_evaled
    # opt_arr[dataset_num] /= num_evaled
    # rand_arr[dataset_num] /= num_evaled
    # compare_opt[dataset_num] /= num_evaled
    # greedy_arr[dataset_num] /= num_evaled
    # print("###################################")
    # # print(acc_arr[dataset_num])
    # # print(opt_arr[dataset_num])

# ax = sns.lineplot(
#     x="name",
#     y="score",
#     hue="Method",
#     style="Method",
#     markers=True,
#     data=stats
# ).set_title("MCTS, Greedy and Random Optimality Ratios over Different Sizes")
# plt.xlabel("Compatibility Graph Size")
# plt.ylabel("Optimality Ratio")
# plt.show()

ax1 = sns.lineplot(
    x="name",
    y="score",
    hue="Method",
    style="Method",
    markers=True,
    data=stats
).set_title("MCTS, Greedy and Random Optimality Ratios over Different Sparsities")
plt.xlabel("Compatibility Graph Sparsity (Edge Probability)")
plt.ylabel("Optimality Ratio")
plt.show()

# print("Avg Acc" + str(acc_arr))
# plt.plot(acc_arr, 'r+')
# plt.title("Performance of {0} trained network over {1}".format(model_name, dataset_name))
# plt.xlabel("Size of testing graph")
# plt.ylabel("Accuracy")
# plt.xticks(np.arange(len(acc_arr)), dataset_names, rotation=90)
# plt.show()
#
# print("Avg Rollout / Optimal Ratio" + str(opt_arr))
# plt.plot(opt_arr, "b+")
# plt.title("Performance of {0} trained network over {1}".format(model_name, dataset_name))
# plt.xlabel("Size of testing graph")
# plt.ylabel("Score Ratio (Avg Rollout / Optimal Score)")
# plt.xticks(np.arange(len(opt_arr)), dataset_names, rotation=90)
# plt.show()
#
# print("Avg Random / Optimal Ratio" + str(rand_arr))
# plt.plot(rand_arr, "b+")
# plt.title("Performance of random")
# plt.xlabel("Size of testing graph")
# plt.ylabel("Score Ratio (Avg Random / Optimal Score)")
# plt.xticks(np.arange(len(rand_arr)), dataset_names, rotation=90)
# plt.show()
#
# print("Random Optimality Bound" + str(compare_opt))
# plt.plot(compare_opt, "b+")
# plt.title("Optimality Ratio - Random Baseline")
# plt.xlabel("Size of testing graph")
# plt.ylabel("Ratio")
# plt.xticks(np.arange(len(compare_opt)), dataset_names, rotation=90)
# plt.show()
#
# print("Avg Greedy OPT" + str(greedy_arr))
# plt.plot(greedy_arr, "b+")
# plt.title("Avg Greedy OPT Ratio")
# plt.xlabel("Size of testing graph")
# plt.ylabel("Ratio")
# plt.xticks(np.arange(len(greedy_arr)), dataset_names, rotation=90)
# plt.show()

# print("Algo, Greedy and Random OPT Ratios" + str(greedy_arr))
# plt.plot(opt_arr, "bo", linewidth=1, linestyle='--', label="GCN")
# plt.plot(greedy_arr, "go", linewidth=1, linestyle='--', label="Greedy")
# plt.plot(rand_arr, "ro", linewidth=1, linestyle='--', label="Random")
# plt.legend()
# plt.title("Algo, Greedy and Random OPT Ratios")
# plt.xlabel("Sparsity (Edge Probability)")
# plt.ylabel("Ratio")
# plt.xticks(np.arange(len(greedy_arr)), dataset_names, rotation=90)
# plt.show()


# print(compare_opt)
# plt.plot(compare_opt, "b+")
# plt.title("Normalized Performance of {0} trained network over {1}".format(model_name, dataset_name))
# plt.xlabel("Size of testing graph")
# plt.ylabel("Optimality")
# plt.xticks(np.arange(len(compare_opt)), dataset_sizes, rotation=90)
# plt.show()
