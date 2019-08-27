import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.nn import BatchNorm1d
from torch_geometric.nn import GCNConv, GraphConv, GATConv, GatedGraphConv

from torch_geometric.utils import add_remaining_self_loops, add_self_loops

class BasicNet(torch.nn.Module):

    def __init__(self):
        super(BasicNet, self).__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, data):
        weight = data["x"].view(-1, 1)
        edge_index = data["edge_index"]
        x = self.conv1(weight, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.sigmoid(x)
        return x


class MultiNet(torch.nn.Module):
    def __init__(self):
        super(MultiNet, self).__init__()
        self.conv_in = GCNConv(1, 32)
        self.conv_mid = GCNConv(32, 32)
        self.conv_out = GCNConv(32, 1)

    def forward(self, data):
        weight = data["x"].view(-1, 1)
        edge_index = add_self_loops(data["edge_index"])
        x = self.conv_in(weight, edge_index)
        x = F.relu(x)

        x = self.conv_mid(x, edge_index)
        x = F.relu(x)

        x = self.conv_mid(x, edge_index)
        x = F.relu(x)

        x = self.conv_out(x, edge_index)
        x = torch.sigmoid(x)
        return x

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_in = GraphConv(1, 32, aggr="add")
        self.conv_max1 = GraphConv(32, 64, aggr="max")
        self.conv_mean1 = GraphConv(64, 32, aggr="mean")

        self.conv_max2 = GraphConv(32, 64, aggr="max")
        self.conv_mean2 = GraphConv(64, 32, aggr="mean")

        self.conv_max3 = GraphConv(32, 64, aggr="max")
        self.conv_mean3 = GraphConv(64, 32, aggr="mean")

        self.conv_out = GraphConv(32, 1, aggr="max")

    def forward(self, data):
        weight = data["x"].view(-1, 1)
        edge_index = data["edge_index"]
        x = self.conv_in(weight, edge_index)
        x = F.relu(x)

        x = self.conv_max1(x, edge_index)
        x = F.relu(x)

        x = self.conv_mean1(x, edge_index)
        x = F.relu(x)

        x = self.conv_max2(x, edge_index)
        x = F.relu(x)

        x = self.conv_mean2(x, edge_index)
        x = F.relu(x)

        x = self.conv_max3(x, edge_index)
        x = F.relu(x)

        x = self.conv_mean3(x, edge_index)
        x = F.relu(x)

        x = self.conv_out(x, edge_index)
        x = torch.sigmoid(x)
        return x

class ConvNet2(torch.nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()
        self.conv_in = GraphConv(1, 32, aggr="add")
        self.conv_mean1 = GraphConv(32, 32, aggr="mean")
        self.conv_mean2 = GraphConv(32, 32, aggr="add")
        self.conv_mean3 = GraphConv(32, 32, aggr="mean")
        self.conv_max3 = GraphConv(32, 32, aggr="max")
        self.conv_out = GraphConv(32, 1, aggr="max")

    def forward(self, data):
        weight = data["x"].view(-1, 1)
        edge_index = data["edge_index"]
        x = self.conv_in(weight, edge_index)
        x = F.relu(x)
        x = self.conv_mean1(x, edge_index)
        x = F.relu(x)
        x = self.conv_mean2(x, edge_index)
        x = F.relu(x)
        x = self.conv_mean3(x, edge_index)
        x = F.relu(x)
        x = self.conv_max3(x, edge_index)
        x = F.relu(x)
        x = self.conv_out(x, edge_index)
        x = torch.sigmoid(x).reshape(-1)
        return x

class ConvNet3(torch.nn.Module):
    def __init__(self):
        super(ConvNet3, self).__init__()
        self.conv_in = GraphConv(1, 32, aggr="add")

        self.conv_mean1 = GraphConv(32, 32, aggr="mean")
        self.conv_max1 = GraphConv(32, 32, aggr="max")

        self.conv_mean2 = GraphConv(32, 32, aggr="add")
        self.conv_max2 = GraphConv(32, 32, aggr="max")

        self.conv_mean3 = GraphConv(32, 32, aggr="mean")
        self.conv_max3 = GraphConv(32, 32, aggr="max")

        self.conv_mean4 = GraphConv(32, 32, aggr="mean")
        self.conv_max4 = GraphConv(32, 32, aggr="max")

        self.conv_mean5 = GraphConv(32, 32, aggr="mean")
        self.conv_max5 = GraphConv(32, 32, aggr="max")

        self.conv_out = GraphConv(32, 1, aggr="max")

    def forward(self, data):
        weight = data["x"].view(-1, 1)
        edge_index, _ = add_self_loops(data["edge_index"], num_nodes=len(data["x"]))

        #edge_index = data["edge_index"]
        x = self.conv_in(weight, edge_index)
        x = F.relu(x)

        x = self.conv_mean1(x, edge_index)
        x = F.relu(x)

        x = self.conv_max1(x, edge_index)
        x = F.relu(x)

        x = self.conv_mean2(x, edge_index)
        x = F.relu(x)

        x = self.conv_max2(x, edge_index)
        x = F.relu(x)

        x = self.conv_mean3(x, edge_index)
        x = F.relu(x)

        x = self.conv_max3(x, edge_index)
        x = F.relu(x)

        x = self.conv_mean4(x, edge_index)
        x = F.relu(x)

        x = self.conv_max4(x, edge_index)
        x = F.relu(x)

        x = self.conv_mean5(x, edge_index)
        x = F.relu(x)

        x = self.conv_max5(x, edge_index)
        x = F.relu(x)

        x = self.conv_out(x, edge_index)
        x = torch.sigmoid(x).reshape(-1)
        return x

