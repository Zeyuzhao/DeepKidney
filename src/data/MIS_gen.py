from data.dataset_gen import *

def generate_MIS(name, num_examples_list, num_nodes_list, edge_prob_list, weighted = False, seed = 42, extreme=False):
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
                        weights = np.random.randint(2, 5, size=num_nodes)
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
    num_size_types = 5
    size_diff = 5

    num_examples_list = [200] * num_size_types
    num_nodes_list = [size_diff * i for i in range(1, num_size_types + 1)]
    edge_prob_list = [10 / (i + 10) for i in num_nodes_list]

    dataset_name = "mini_MIS"
    if osp.exists("../../data/" + dataset_name):
        raise Exception("Existing dataset has the name [{0}]. Either remove the existing dataset, or rename the "
                        "current one".format(dataset_name))

     # Generate in one combined:
    generate_MIS(name=dataset_name,
                     num_examples_list=num_examples_list,
                     num_nodes_list=num_nodes_list,
                     edge_prob_list=edge_prob_list,
                     weighted=True,
                     seed=900)


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