

import os.path as osp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import torch
from torch_geometric.data import InMemoryDataset, DataLoader, Data
from torch_geometric.utils import to_networkx

from env.Env import edgeToAdj, MaxSetState

def from_networkx(G):
    r""""Converts a networkx graph to a data object graph.

    Args:
        G (networkx.Graph): A networkx graph.
    """

    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    data = {}
    data['edge_index'] = edge_index
    data = Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()
    return data

class MaxIndDataset(InMemoryDataset):
    def __init__(self, root):
        print(osp.abspath(osp.join(root, "weight.csv")))
        self.label_frame = pd.read_csv(osp.join(root, "label.csv"), engine='python')
        self.weight_frame = pd.read_csv(osp.join(root, "weight.csv"), engine='python')

        self.root_dir = root
        self.num_graphs = len(self.label_frame.index)

        #print(self.num_graphs)
        super(MaxIndDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def _download(self):
        pass

    def process(self):
        data_list = []
        for i in range(self.num_graphs):
            graph_name = osp.join(self.root_dir, self.label_frame.iloc[i, 0])
            graph = nx.read_adjlist(graph_name, nodetype=int)

            weight = torch.tensor(self.weight_frame.iloc[i, 1:].dropna(), dtype=torch.float)
            label = torch.tensor(self.label_frame.iloc[i, 1:].dropna(), dtype=torch.long)

            data = from_networkx(graph)

            # Handle case of no edges, the program breaks down
            if len(data['edge_index']) == 0:
                data["edge_index"] = torch.tensor([[], []], dtype=torch.long)
            data.x = weight
            data.y = label
            #print("{0}: {1}".format(len(weight), data["edge_index"]))
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def state_repr(self, index):
        item = self[index]
        np_item = {k: np.round(v.cpu().data.numpy(), 3) for k, v in iter(item)}
        np_item["adj_matrix"] = edgeToAdj(len(np_item["x"]), np_item["edge_index"])
        return MaxSetState(np_item)

def split_loader(dataset, train_size, test_size, batch_size):
    dataset.shuffle()
    size = len(dataset)

    tr_i = int(size * train_size)
    val_i = tr_i + int(size * test_size)
    train_loader = DataLoader(dataset[:tr_i], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset[tr_i: val_i])
    test_loader = DataLoader(dataset[val_i:])
    return train_loader, val_loader, test_loader

def draw_entry(entry, node_color=None, edge_color = None, title=None, node_dict=None, cmap = None):
    g = to_networkx(entry)

    border_cmap = ["black", "blue"]
    if not cmap:
        cmap = ["red", "green"]

    if node_color is not None:
        label = node_color
    else:
        label = entry["y"]

    # Create color map from selected nodes, green for selected, grey for unselected.
    node_color = [cmap[0]] * len(g.nodes)
    border_color = [border_cmap[0]] * len(g.nodes)

    for i in np.flatnonzero(label):
        node_color[i] = cmap[1]

    for i in np.flatnonzero(edge_color):
        border_color[i] = border_cmap[1]


    node_labels = entry["x"]

    if node_dict is None:
        node_dict = {v: v for v in range(len(node_labels))}

    if not torch.equal(node_labels, torch.ones(len(g.nodes))):
        node_labels = {k: "{0}:\n{1:.3f}".format(node_dict[k], v) for (k, v) in enumerate(node_labels)}
    else:
        node_labels = {k: node_dict[k] for k in g.nodes}

    plt.figure()
    plt.title(title)
    pos = nx.circular_layout(g)

    print(border_color)
    nx.draw(g, pos, node_size=2000, width=1, linewidths=3, node_color = node_color, edgecolors = border_color)
    nx.draw_networkx_labels(g, pos, node_labels)
    plt.show()


if __name__ == '__main__':
    dataset = MaxIndDataset("../../data/weighted_mix")
    print(len(dataset))
    for i in range(0, 5):
        print(dataset[i])
        draw_entry(dataset[i])
