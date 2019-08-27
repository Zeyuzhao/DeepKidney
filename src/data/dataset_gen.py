import json
import os
import warnings
import tarfile
import numpy as np
import csv
import matplotlib.pyplot as plt
import networkx as nx
from gurobipy import *
from tqdm import tqdm

warnings.filterwarnings("ignore")

import random
def draw_reduced(g, node_color=None):
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
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def generate_dataset(name, num_examples_list, num_nodes_list, edge_prob_list, weighted = False, seed = 42, extreme=False):
    # Get the parameters of the function, need to be first
    params = locals()
    random.seed(seed)
    np.random.seed(seed)
    root_dir = "../../data/" + name
    label_filename = "label.csv"
    weight_filename = 'weight.csv'
    param_filename = "param.json"
    os.makedirs(os.path.join(root_dir), exist_ok=True)
    with open(os.path.join(root_dir, param_filename), 'w+') as param_file:
        json.dump(params, param_file)
    with open(os.path.join(root_dir, label_filename), 'w+') as label_file:
        with open(os.path.join(root_dir, weight_filename), 'w+') as weight_file:
            label_writer = csv.writer(label_file, delimiter=',')
            weight_writer = csv.writer(weight_file, delimiter=',')

            max_size = max(num_nodes_list)
            label_writer.writerow(["Filename"] + list(range(max_size)))
            weight_writer.writerow(["Filename"] + list(range(max_size)))

            for nth, num_nodes in enumerate(num_nodes_list):
                for j in tqdm(range(num_examples_list[nth])):
                    graph_filename = "graph_{0}_{1}.txt".format(num_nodes, j)
                    graph = nx.fast_gnp_random_graph(num_nodes, edge_prob_list[nth], directed=False)

                    if weighted:
                        weights = np.random.randint(1, 5, size=num_nodes)
                        if extreme:
                            index = np.random.randint(num_nodes)
                            weights[index] = 10
                    else:
                        weights = np.ones(num_nodes)

                    for v in graph.nodes():
                        graph.nodes[v]['weight'] = weights[v]

                    with open(os.path.join(root_dir, graph_filename), 'wb+') as graph_file:
                        nx.write_adjlist(graph, graph_file)

                    output_list = compute_max_ind_set(graph)

                    output = np.zeros(num_nodes, dtype=int)
                    output[output_list] = 1
                    label_writer.writerow([graph_filename] + list(output))
                    weight_writer.writerow([graph_filename] + list(weights))
                    label_file.flush()
    generate_tarfile(root_dir + '.tar.gz', root_dir)
    print("done")

if __name__ == '__main__':
    num_size_types = 10
    size_diff = 10


    num_examples_list = [500] * 20
    num_nodes_list = [size_diff * i for i in range(1, num_size_types + 1)]
    edge_prob_list = [3 / (i + 3) for i in num_nodes_list]

    # Generate in one combined:
    # generate_dataset(name="med_weighted_mix",
    #                  num_examples_list=num_examples_list,
    #                  num_nodes_list=num_nodes_list,
    #                  edge_prob_list=edge_prob_list,
    #                  weighted=True,
    #                  seed=42)


    # Generate in distinct
    for i in range(len(num_nodes_list)):
        num_nodes = num_nodes_list[i]
        edge_prob = edge_prob_list[i]

        generate_dataset(name="int_42/weighted_{0}".format(num_nodes),
                         num_examples_list=[1000],
                         num_nodes_list=[num_nodes],
                         edge_prob_list=[edge_prob],
                         weighted=True,
                         seed=42,
                         extreme=False)