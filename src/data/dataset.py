import json
import os.path as osp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import torch
from torch_geometric.data import InMemoryDataset, DataLoader, Data
from torch_geometric.utils import to_networkx

from env.Env import edgeToAdj, MaxSetState

import ast


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


MAX_CYCLE_LEN = 3


class KidneyDataset(InMemoryDataset):
    """
    A KidneyDataset loads Compatibility Graphs.

    Loads the dataset_gen.py generate_kidney() output.
    """
    def __init__(self, root):

        # print(osp.abspath(osp.join(root, "weight.csv")))
        self.max_size = int(open(osp.join(root, "size.txt"), 'r').read())
        self.label_frame = pd.read_csv(osp.join(root, "label.csv"), engine='python',
                                       names=["Filename"] + list(range(self.max_size)))
        self.weight_frame = pd.read_csv(osp.join(root, "weight.csv"), engine='python',
                                        names=["Filename"] + list(range(self.max_size)))
        self.cyc_frame = pd.read_csv(osp.join(root, "cycles.csv"), engine='python',
                                     names=["Filename"] + list(range(self.max_size)))
        self.root_dir = root
        self.num_graphs = len(self.label_frame.index)

        # with open(osp.join(root, "cycles.csv")) as json_file:
        #     data = json.load(json_file)
        # print(self.num_graphs)
        super(KidneyDataset, self).__init__(root)
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
            cyc_graph = nx.read_adjlist(graph_name, nodetype=int)

            comp_graph_name = osp.join(self.root_dir, self.label_frame.iloc[i, 0].replace("cyc", "comp"))
            comp_graph = nx.read_adjlist(comp_graph_name, nodetype=int, create_using=nx.DiGraph())

            weight = torch.tensor(self.weight_frame.iloc[i, 1:].dropna(), dtype=torch.float)
            label = torch.tensor(self.label_frame.iloc[i, 1:].dropna(), dtype=torch.long)

            # We pad the cycles tensor (with -1) because tensors must have uniform matrices; no jagged arrays
            cycles = torch.tensor([self.pad_arr(ast.literal_eval(c), MAX_CYCLE_LEN)
                                   for c in self.cyc_frame.iloc[i, 1:].dropna()],
                                  dtype=torch.long)

            data = from_networkx(cyc_graph)
            # Handle case of no edges, the program breaks down
            if len(data['edge_index']) == 0:
                data["edge_index"] = torch.tensor([[], []], dtype=torch.long)

            data["comp_index"] = torch.tensor(list(comp_graph.edges)).t().contiguous()
            data["num_comp_nodes"] = torch.max(data["comp_index"]).item() + 1

            data.x = weight
            data.y = label
            data["cycle"] = cycles
            # print("{0}: {1}".format(len(weight), data["edge_index"]))
            data_list.append(data)
        pass
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def pad_arr(self, arr, n):
        return arr + [-1] * max(n - len(arr), 0)

    def state_repr(self, index):
        item = self[index]
        np_item = {k: np.round(v.cpu().data.numpy(), 3) for k, v in iter(item)}
        np_item["adj_matrix"] = edgeToAdj(len(np_item["x"]), np_item["edge_index"])
        return MaxSetState(np_item)


class MaxIndDataset(InMemoryDataset):
    """
    A KidneyDataset loads Compatibility Graphs

    Generates custom file structure for storing MWIS graphs' adjacency list, weights and an optimal solution.
    """
    def __init__(self, root):
        # print(osp.abspath(osp.join(root, "weight.csv")))
        self.label_frame = pd.read_csv(osp.join(root, "label.csv"), engine='python')
        self.weight_frame = pd.read_csv(osp.join(root, "weight.csv"), engine='python')

        self.root_dir = root
        self.num_graphs = len(self.label_frame.index)

        # print(self.num_graphs)
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
            # print("{0}: {1}".format(len(weight), data["edge_index"]))
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def state_repr(self, index):
        item = self[index]
        np_item = {k: np.round(v.cpu().data.numpy(), 3) for k, v in iter(item)}
        np_item["adj_matrix"] = edgeToAdj(len(np_item["x"]), np_item["edge_index"])
        return MaxSetState(np_item)


# Splits dataset into training and testing sections.
# Batch size indicates the number of graphs aggregated into one graph.
def split_loader(dataset, train_size, test_size, batch_size):
    dataset.shuffle()
    size = len(dataset)

    tr_i = int(size * train_size)
    val_i = tr_i + int(size * test_size)
    train_loader = DataLoader(dataset[:tr_i], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset[tr_i: val_i])
    test_loader = DataLoader(dataset[val_i:])
    return train_loader, val_loader, test_loader

#
def draw_MIS(entry, node_color=None, edge_color=None, title=None, node_dict=None, cmap=None):
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

    # print(border_color)
    nx.draw(g, pos, node_size=2000, width=1, linewidths=3, node_color=node_color, edgecolors=border_color)
    nx.draw_networkx_labels(g, pos, node_labels)

    plt.show()


def draw_comp(entry, title=None):
    G = nx.DiGraph()
    G.add_nodes_from(range(entry["num_comp_nodes"]))
    for i, (u, v) in enumerate(entry["comp_index"].t().tolist()):
        G.add_edge(u, v)

    included = set()
    edge_list = list_to_edge(list(entry["cycle"][entry["y"].to(bool)]))

    for c in edge_list:
        nodes = np.array(c).flatten()
        print(nodes)
        included.update(list(nodes))
    cmap = ["red", "green"]

    node_color = np.array([cmap[0]] * len(G.nodes), dtype='object')

    for n in included:
        node_color[n] = "green"

    plt.figure()
    plt.title(title)
    pos = nx.circular_layout(G)

    # print(border_color)c
    node_labels = {k: k for k in G.nodes}
    nx.draw(G, pos, node_size=2000, width=1, linewidths=3, node_color=node_color)
    nx.draw_networkx_labels(G, pos, node_labels)

    for e in edge_list:
        nx.draw_networkx_edges(G, pos,
                               edgelist=e,
                               width=8, alpha=0.5, edge_color='r')
    plt.show()


def draw_all_cyc(entry, title=None):
    G = nx.DiGraph()
    G.add_nodes_from(range(entry["num_comp_nodes"]))
    for i, (u, v) in enumerate(entry["comp_index"].t().tolist()):
        G.add_edge(u, v)

    edge_list = list_to_edge(list(entry["cycle"]))

    for e in edge_list:
        plt.figure()
        plt.title(title)
        pos = nx.circular_layout(G)
        # print(border_color)c
        node_labels = {k: k for k in G.nodes}
        nx.draw(G, pos, node_size=1000, width=1, linewidths=3)
        nx.draw_networkx_labels(G, pos, node_labels)
        nx.draw_networkx_edges(G, pos,
                               edgelist=e,
                               width=8, alpha=0.5, edge_color='r')
        plt.show()


def loop_to_edges(arr):
    out = []
    arr = arr[arr >= 0].cpu().data.numpy()
    for i in range(len(arr) - 1):
        out.append((arr[i], arr[i + 1]))
    out.append((arr[-1], arr[0]))
    return out


def list_to_edge(arr):
    total_edges = []
    for l in arr:
        total_edges.append(loop_to_edges(l))
    return total_edges


if __name__ == '__main__':
    dataset = KidneyDataset("../../data/cyc_2/weighted_100")
    print(len(dataset))
    # for i in range(0, 5):
    #     print(dataset[i])
    #     draw_MIS(dataset[i])
    draw_comp(dataset[0])
    # draw_all_cyc(dataset[0])
