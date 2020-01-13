from model.GCN import *
from data.dataset import *
from collections import OrderedDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

conv_net = ConvNet3().to(device)

model_folder= "../../model/run_1/"
model_name= "size_mixed"
model_file = "epoch_9.pt"

params = torch.load(osp.join(model_folder, model_name, model_file))
conv_net.load_state_dict(params)
conv_net.eval()

visualize = KidneyDataset("../../data/mini_MIS")

for ID in range(0, len(visualize), 100):
    test_item = visualize[ID]

    # Testing extreme cases
    #test_item.x[5] = 100

    test_item.to(device)
    out = conv_net(test_item)

    # Sorts the indices from greatest value to the least

    vals = out.detach().cpu().numpy().T.flatten()
    labels = test_item["y"].detach().cpu().numpy().T.flatten()
    print(vals)

    bool_vals = vals >= 0.5

    correct = np.equal(bool_vals, labels)

    rankings = np.flip(vals.argsort())
    dictionary = OrderedDict()

    draw_MIS(test_item.to(torch.device("cpu")), title="weighted_10: " + str(ID), node_color=correct, edge_color=labels)

    print("ID: " + str(ID))
    print("Model Rankings: {0}".format(rankings))
    for i in range(len(rankings)):
        dictionary[rankings[i]] = vals[rankings[i]]
    print(dictionary)
    print("**************************************")

