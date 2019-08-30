from model.GCN import *
from data.dataset import *
from collections import OrderedDict
from tqdm import tqdm
from glob import glob
from model.EvalUtils import *

DEBUG = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

conv_net = ConvNet3().to(device)



model_folder= "../../model/run_1/"
model_name= "size_mixed"
model_file = "epoch_9.pt"

params = torch.load(osp.join(model_folder, model_name, model_file))
conv_net.load_state_dict(params)
conv_net.eval()


dataset_name = "binary"
dataset_folder = osp.join("../../data", dataset_name)

dataset_subfolders = glob(osp.join(dataset_folder, "[!processed]*/"))

print(dataset_subfolders)

# Sorts the folder by getting the graph's sizes

# Format of each subfolder: [name]_10/
dataset_subfolders.sort(key=lambda x: int(x.split("_")[-1][:-1]) if "_" in x else 0)

if len(dataset_subfolders) == 0:
    dataset_subfolders.append(dataset_folder)

acc_arr = [0] * len(dataset_subfolders)
opt_arr = [0] * len(dataset_subfolders)
dataset_sizes = []

for dataset_num, dataset_dir in enumerate(dataset_subfolders):
    test_set = MaxIndDataset(dataset_dir)
    for ID in tqdm(range(len(test_set))):
        test_item = test_set[ID]
        if ID == 0:
            dataset_sizes.append(len(test_item["x"]))
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
        score = 0
        for i in range(5):
            s, actions = max_eval(test_item, conv_net)
            score = max(s, score)


        # Get the sum of weights dictated by weighting
        label_score = sum(test_item["x"][test_item["y"].to(torch.bool)]).item()
        opt_percentage = score / label_score
        acc_arr[dataset_num] += acc
        opt_arr[dataset_num] += opt_percentage

        if DEBUG and acc < 0.99:
            # Sorts the indices from greatest value to the least
            rankings = np.flip(vals.argsort())
            dictionary = OrderedDict()
            #draw_entry(test_item.to(torch.device("cpu")), title="weighted_10: " + str(ID), node_color=correct, edge_color=labels)
            print("ID: " + str(ID))
            print("Model Rankings: {0}".format(rankings))
            for i in range(len(rankings)):
                dictionary[rankings[i]] = vals[rankings[i]]
            print(dictionary)
            print(vals)
            print(correct)
            print("Weights: " + str(test_item["x"].cpu().data.numpy()))
            print("GNN: " + str(score))
            print("Optimal: " + str(label_score))
            print("**************************************")
            input()
    acc_arr[dataset_num] /= len(test_set)
    opt_arr[dataset_num] /= len(test_set)
    print("###################################")
    print(acc_arr[dataset_num])
    print(opt_arr[dataset_num])

print(acc_arr)
plt.plot(acc_arr, 'r+')
plt.title("Performance of {0} trained network over {1}".format(model_name, dataset_name))
plt.xlabel("Size of testing graph")
plt.ylabel("Accuracy")
plt.xticks(np.arange(len(acc_arr)), dataset_sizes, rotation=90)
plt.show()

print(opt_arr)
plt.plot(opt_arr, "b+")
plt.title("Performance of {0} trained network over {1}".format(model_name, dataset_name))
plt.xlabel("Size of testing graph")
plt.ylabel("Optimality")
plt.xticks(np.arange(len(opt_arr)), dataset_sizes, rotation=90)
plt.show()
