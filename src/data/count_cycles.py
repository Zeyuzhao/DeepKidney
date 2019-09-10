
# James Strimble
from collections import deque

from networkx import DiGraph, path_graph
import networkx as nx
import matplotlib.pyplot as plt


def count_cycles(digraph: DiGraph, max_length):
    """Find cycles of length up to max_length in the digraph.
    Return value: a list of cycles, in which each cycle is represented
    as a list of vertices, with the first vertex not repeated.
    """

    counts_by_size = [0] * (max_length + 1)

    if max_length < 2:
        return counts_by_size

    vtx_used = [False] * len(digraph.nodes)
    #cycle_arr = [[] for _ in range(max_length + 1)]
    cycle_arr = []
    def cycle(first_vtx, last_vtx, num_vertices, vtx_arr):
        if digraph.has_edge(last_vtx, first_vtx):
            counts_by_size[num_vertices] += 1
            #cycle_arr[num_vertices].append(vtx_arr)
            cycle_arr.append(vtx_arr)
        if num_vertices < max_length:
            for v in digraph.neighbors(last_vtx):
                if (num_vertices + shortest_paths_to_low_vtx[v] <= max_length
                            and not vtx_used[v]):
                    vtx_used[v] = True
                    cycle(first_vtx, v, num_vertices + 1, vtx_arr + [v])
                    vtx_used[v] = False

    # Adjacency lists for transpose graph
    transp_adj_lists = [[] for v in digraph.nodes]
    for edge in digraph.edges:
        transp_adj_lists[edge[1]].append(edge[0])

    for v in digraph.nodes:
        shortest_paths_to_low_vtx = calculate_shortest_path_lengths(
                digraph,
                v,
                max_length - 1,
                lambda u: (w for w in transp_adj_lists[u] if w > v)
        )
        vtx_used[v] = True
        cycle(v, v, 1, [v])
        vtx_used[v] = False
    return cycle_arr

def calculate_shortest_path_lengths(digraph: DiGraph, from_v, max_dist,
                                    adj_list_accessor):
    """Calculate the length of the shortest path from vertex from_v to each
    vertex with a greater or equal index, using paths containing
    only vertices indexed greater than or equal to from_v.
    Return value: a list of distances of length equal to the number of vertices.
    If the shortest path to a vertex is greater than max_dist, the list element
    will be 999999999.
    Args:
        from_v: The starting vertex
        max_dist: The maximum distance we're interested in
        adj_list_accessor: A function taking a vertex and returning an
            iterable of out-edge targets
    """
    # Breadth-first search
    q = deque([from_v])
    distances = [999999999] * len(digraph.nodes)
    distances[from_v] = 0

    while q:
        v = q.popleft()
        # Note: >= is used instead of == on next line in case max_dist<0
        if distances[v] >= max_dist:
            break
        adj = list(adj_list_accessor(v))
        for w in adj:
            if distances[w] == 999999999:
                distances[w] = distances[v] + 1
                q.append(w)
    return distances

def cycle_graph(n):
    """Return the cycle graph C_n over n nodes.

    C_n is the n-path with two end-nodes connected.

    Node labels are the integers 0 to n-1
    If create_using is a DiGraph, the direction is in increasing order.

    """
    G=path_graph(n, create_using=DiGraph)
    G.name="cycle_graph(%d)"%n
    if n > 1: G.add_edge(n - 1, 0)
    return G

if __name__ == '__main__':
    cycle = cycle_graph(4)
    cycle.add_edge(1, 0)
    pos = nx.circular_layout(cycle)
    nx.draw(cycle, pos)
    nx.draw_networkx_labels(cycle, pos)
    plt.show()
    print(count_cycles(cycle, 2)[0])