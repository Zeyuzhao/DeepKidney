from model.GCN import *
from data.dataset import *
from collections import OrderedDict


DEBUG = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

conv_net = ConvNet3().to(device)

name="weighted_convnet_epoch_9.pt"
folder="wandb/run-20190827_013223-jkt02lp5"

params = torch.load(osp.join(folder, name))
conv_net.load_state_dict(params)
conv_net.eval()

acc_arr = [0] * 10
for i, num_nodes in enumerate(range(10, 101, 10)):
    test_set = MaxIndDataset("../../data/int_42/weighted_{0}".format(num_nodes))
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

        acc_arr[i] += acc
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
    acc_arr[i] /= len(test_set)

