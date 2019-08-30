from model.GCN import *
from data.dataset import *
from collections import OrderedDict
from tqdm import tqdm

DEBUG = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

acc_matrix = np.zeros((10, 10))

#Load datasets into an array
dataset_arr = []
for i, num_nodes in enumerate(range(10, 101, 10)):
    dataset_arr.append(MaxIndDataset("../../data/testset_43/weighted_{0}".format(num_nodes)))

for a, gcn_size in enumerate(range(10, 101, 10)):
    conv_net = ConvNet3().to(device)
    name = "epoch_9.pt"
    folder = "../../model/run_1/size_{0}".format(a)
    params = torch.load(osp.join(folder, name))
    conv_net.load_state_dict(params)
    conv_net.eval()

    for i, num_nodes in enumerate(tqdm(range(10, 101, 10))):
        test_set = dataset_arr[i]
        for ID in range(len(test_set)):
            test_item = test_set[ID]

            test_item.to(device)
            out = conv_net(test_item)


            vals = out.detach().cpu().numpy().T.flatten()
            labels = test_item["y"].detach().cpu().numpy().T.flatten()
            #print(vals)

            bool_vals = vals >= 0.5
            correct = np.equal(bool_vals, labels)

            acc = np.count_nonzero(correct) / len(correct)

            acc_matrix[a, i] += acc
            if DEBUG:
                # Sorts the indices from greatest value to the least
                rankings = np.flip(vals.argsort())
                dictionary = OrderedDict()
                draw_entry(test_item.to(torch.device("cpu")), title="weighted_10: " + str(ID), node_color=correct, edge_color=labels)
                print("ID: " + str(ID))
                print("Model Rankings: {0}".format(rankings))
                for i in range(len(rankings)):
                    dictionary[rankings[i]] = vals[rankings[i]]
                print(dictionary)
                print("**************************************")
        acc_matrix[a][i] /= len(test_set)
        #print(acc_matrix[a][i])
    print(acc_matrix[a])

ax = plt.plot(acc_matrix.transpose())
plt.title("Performance of different size networks over varying size graphs")
plt.xlabel("Size of testing graph")
plt.ylabel("Accuracy")
plt.legend(["gcn_{0}".format((i + 1) * 10) for i in range(10)], bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

