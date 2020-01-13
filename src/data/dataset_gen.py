import json
import os
import warnings
import tarfile
import numpy as np
import csv
import matplotlib.pyplot as plt
import networkx as nx
from gurobipy import *
from networkx import DiGraph
from tqdm import tqdm
import os.path as osp

from data.count_cycles import count_cycles

warnings.filterwarnings("ignore")

import random
def draw_reduced(g, node_color=None):
    """
    Displays a visual representation of the graph
    :param g: Input graph G
    :param node_color: The color of the highlighted nodes
    :return: None
    """
    node_lables = nx.get_node_attributes(g, "weight")

    if node_lables:
        node_lables = {k: "{0}: {1}".format(k, v) for (k, v) in node_lables.items()}
        # if not node_color:
        #     node_color = node_lables.
    else:
        node_lables = {k: k for k in g.nodes}

    plt.figure()
    pos = nx.circular_layout(g)
    nx.draw(g, pos, node_size=2000, width=1, node_color=node_color)
    nx.draw_networkx_labels(g, pos, node_lables)
    plt.show()
    # plt.savefig("graph.png", dpi=1000)


def compute_max_ind_set(graph, display=False, debug=False):
    """
    Computes the MWIS from a specified graph
    :param graph: The input graph, either weighted or unweighted
    :param display: Flag for displaying
    :param debug: Flag for debugging
    :return: None
    """
    nodes = list(graph.nodes)
    model = Model('Maximum Independent Set')

    # Suppress output
    if not debug:
        model.setParam('OutputFlag', 0)

    # An indicator variable for whether a node is chosen
    indicators = model.addVars(nodes, vtype=GRB.BINARY, name="x")

    # Set Constraint: If two nodes are bound by an edge, they cannot be both chosen
    model.addConstrs(((indicators[i] + indicators[j] <= 1) for (i, j) in graph.edges), "Max")

    # Set objective: Maximize the weighted sum of nodes, or if no weights, just the cardinality.
    weights = nx.get_node_attributes(graph, "weight")

    if weights:
        obj = sum([indicators[i] * weights[i] for i in nodes])
    else:
        obj = sum([indicators[i] for i in nodes])

    model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()

    # Find all selected nodes and save them in soln
    soln = []
    for i in range(len(indicators)):
        if indicators[i].x != 0:
            soln.append(i)

    if display:
        # Create color map from selected nodes, green for selected, grey for unselected.
        color_map = ["grey"] * len(graph.nodes)
        for i in soln:
            color_map[i] = "green"
        draw_reduced(graph, color_map)
    return soln


def generate_tarfile(output_filename, source_dir):
    """
    Create compressed tar.gz file
    :param output_filename: The output filename
    :param source_dir: The source directory
    :return:
    """
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def comp_to_cycle(comp_graph: DiGraph, max_length = 3):
    """
    Generates a new MWIS from a Compatibility Graph
    :param comp_graph: Original Compatibility Graph
    :param max_length: Maximum Length of Cycles (default: 3)
    :return: A new MWIS graph
    """
    cycles = count_cycles(comp_graph, max_length)
    out_graph = nx.DiGraph()
    for i, c in enumerate(cycles):
        out_graph.add_node(i, weight = len(c), cyc = c)

    for i, c in enumerate(cycles):
        for j, d in enumerate(cycles):
            if c is not d and not set(c).isdisjoint(d):
                out_graph.add_edge(i, j)
    return out_graph


def generate_kidney(name, num_examples_list, num_nodes_list, edge_prob_list, weighted = False, seed = 42, extreme=False, debug=False):
    """
    Generates a Kidney Exchange Dataset with various parameters.
    :param name: Filename of the dataset
    :param num_examples_list: Number of graphs for each category
    :param num_nodes_list: Graph size for each category
    :param edge_prob_list: Edge probabilities for each category
    :param weighted: If True, generate random edge probabilities
    :param seed: Seed for random graph generation
    :param extreme:
    :param debug:
    :return:
    """
    # Get the parameters of the function, need to be first
    params = locals()
    random.seed(seed)
    np.random.seed(seed)
    root_dir = "../../data/" + name
    label_filename = "label.csv"
    weight_filename = 'weight.csv'
    cyc_filename = 'cycles.csv'
    param_filename = "param.json"
    size_filename = "size.txt"

    os.makedirs(os.path.join(root_dir), exist_ok=True)
    with open(os.path.join(root_dir, param_filename), 'w+') as param_file:
        json.dump(params, param_file)

    max_cycle_graph_size = 0
    with open(os.path.join(root_dir, label_filename), 'w+') as label_file:
        with open(os.path.join(root_dir, weight_filename), 'w+') as weight_file:
            with open(os.path.join(root_dir, cyc_filename), 'w+') as cyc_file:
                label_writer = csv.writer(label_file, delimiter=',')
                weight_writer = csv.writer(weight_file, delimiter=',')
                cyc_writer = csv.writer(cyc_file, delimiter=',')

                # Flawed!!! only max_size for compat graph, not the cycle graph
                # label_writer.writerow(["Filename"])
                # weight_writer.writerow(["Filename"])
                # cyc_writer.writerow(["Filename"])

                for nth, num_nodes in enumerate(num_nodes_list):
                    for j in tqdm(range(num_examples_list[nth])):
                        comp_graph_filename = "comp_graph_{0}_{1}.txt".format(num_nodes, j)
                        cyc_graph_filename = "cyc_graph_{0}_{1}.txt".format(num_nodes, j)

                        comp_graph = nx.fast_gnp_random_graph(num_nodes, edge_prob_list[nth], directed=True)

                        with open(os.path.join(root_dir, comp_graph_filename), 'wb+') as graph_file:
                            nx.write_adjlist(comp_graph, graph_file)

                        # pos = nx.circular_layout(comp_graph)
                        # nx.draw(comp_graph, pos)
                        # nx.draw_networkx_labels(comp_graph, pos)
                        # plt.show()

                        cyc_graph = comp_to_cycle(comp_graph)
                        max_cycle_graph_size = max(max_cycle_graph_size, len(cyc_graph.nodes))
                        with open(os.path.join(root_dir, cyc_graph_filename), 'wb+') as graph_file:
                            nx.write_adjlist(cyc_graph, graph_file)

                        output_list = compute_max_ind_set(cyc_graph, debug=debug)
                        output = np.zeros(len(cyc_graph.nodes), dtype=int)
                        output[output_list] = 1

                        cycles = nx.get_node_attributes(cyc_graph, "cyc")

                        weights = nx.get_node_attributes(cyc_graph, "weight")

                        label_writer.writerow([cyc_graph_filename] + list(output))
                        # Converting dictionary to list


                        # w_l = []
                        # for i in range(len(weights.keys())):
                        #     w_l = weights[i]
                        weight_writer.writerow([cyc_graph_filename] + list(weights.values()))
                        cyc_writer.writerow([cyc_graph_filename] + list(cycles.values()))

                        label_file.flush()
                        weight_file.flush()
                        cyc_file.flush()

    with open(os.path.join(root_dir, size_filename), 'w+') as size_file:
        size_file.write(str(max_cycle_graph_size))
    generate_tarfile(root_dir + '.tar.gz', root_dir)
    print("done")


if __name__ == '__main__':

    dataset_name = "kidney_fig_sparsity"
    if osp.exists("../../data/" + dataset_name):
        raise Exception("Existing dataset has the name [{0}]. Either remove the existing dataset, or rename the "
                        "current one".format(dataset_name))

    num_nodes_list = [500, 500, 500, 500]
    edge_prob_list = [0.016, 0.018, 0.020, 0.022]
    num_examples_list = [40, 40, 40, 40]
    for i in range(len(num_nodes_list)):
        num_nodes = num_nodes_list[i]
        edge_prob = edge_prob_list[i]
        num_examples = num_examples_list[0]
        generate_kidney(name="{0}/weighted_n{1}_p{2}".format(dataset_name, num_nodes, int(edge_prob * 1000)),
                        num_examples_list=[num_examples],
                        num_nodes_list=[num_nodes],
                        edge_prob_list=[edge_prob],
                        weighted=True,
                        seed=42,
                        extreme=False,
                        debug=False)




    # num_size_types = 1
    # size_diff = 80
    #
    # num_examples_list = [200] * num_size_types
    # num_nodes_list = [size_diff * i for i in range(1, num_size_types + 1)]
    # edge_prob_list = [30 / (i + 30) for i in num_nodes_list]
    #
    # dataset_name = "size_80"
    # if osp.exists("../../data/" + dataset_name):
    #     raise Exception("Existing dataset has the name [{0}]. Either remove the existing dataset, or rename the "
    #                     "current one".format(dataset_name))
    #
    #  # Generate in one combined:
    # generate_dataset(name=dataset_name,
    #                  num_examples_list=num_examples_list,
    #                  num_nodes_list=num_nodes_list,
    #                  edge_prob_list=edge_prob_list,
    #                  weighted=True,
    #                  seed=900)


    # Generate in distinct

    # for i in range(len(num_nodes_list)):
    #     num_nodes = num_nodes_list[i]
    #     edge_prob = edge_prob_list[i]
    #     num_examples = num_examples_list[0]
    #
    #     generate_dataset(name="{0}/weighted_{1}".format(dataset_name, num_nodes),
    #                      num_examples_list=[num_examples],
    #                      num_nodes_list=[num_nodes],
    #                      edge_prob_list=[edge_prob],
    #                      weighted=True,
    #                      seed=10101,
    #                      extreme=False)