import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch_geometric.data import DataLoader

from . import GCN

args = {
    'epochs': 10,
    'cuda': torch.cuda.is_available(),
}

# Input graph dict:
# Contains attributes "x" and "edge_index", each array is a numpy array
# Ex: "x": [1, 1, 1] and "edge_index": [[0, 1], [1, 2]]

class MaximalNetWrapper():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nnet = GCN.ConvNet3().to(self.device)

    # Data comes as a set of examples.
    # Each example is a input graph dictionary
    def train(self, examples):
        optimizer = optim.Adam(self.nnet.parameters())
        for epoch in range(args["epoch"]):
            print('Epoch: ' + str(epoch))
            self.nnet.train()

            # Non-batch method
            for i, item in enumerate(examples):
                graph = {}
                graph["x"] = torch.FloatTensor(item["x"])
                graph["edge_index"] = torch.LongTensor(item["edge_index"])
                node_values = torch.FloatTensor(item["y"])

                if torch.cuda.is_available():
                    graph["x"], graph["edge_index"], node_values = graph["x"].to(self.device), graph["edge_index"].to(self.device), node_values.to(self.device)

                out_pi = self.nnet(graph)
                l_pi = self.loss_pi(node_values, out_pi)
                total_loss = l_pi
                # Todo: record losses using logger
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()


    # Each item is a dictionary with x and edge_index
    def predict(self, item):
        graph = {}
        graph["x"] = torch.FloatTensor(item["x"]).cuda()
        graph["edge_index"] = torch.LongTensor(item["edge_index"]).cuda()

        self.nnet.eval()
        pi = self.nnet(graph)
        return pi.data.cpu().numpy()

    def sampleSolution(self):
        pass

    def loss_pi(self, returns, outputs):
        return -torch.sum(torch.log(outputs) * returns)[0]

    def save_checkpoint(self, folder="model", name="chkpoint.pt"):
        fp = osp.join(folder, name)
        if not osp.exists(folder):
            print("Specified checkpoint directory does not exist! Making directory [{}].".format(folder))
            os.mkdir(folder)
        torch.save({
            'state_dict': self.nnet.state_dict()
        }, fp)

    def load_checkpoint(self, folder="../../model", name="chkpoint.pt"):
        fp = osp.join(folder, name)
        print(fp)
        if not osp.exists(fp):
            raise("No model found in path {}".format(fp))
        chkpoint = torch.load(fp)
        self.nnet.load_state_dict(chkpoint)