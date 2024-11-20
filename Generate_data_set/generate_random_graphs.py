import networkx as nx
import numpy as np
from Generate_data_set.adjacency_matrix_encoding import adjacency_matrix_to_string

def is_excluded_graph(G):
    """
    Check if a graph G matches any excluded pattern:
    - Complete graph
    - Cyclic graph
    - Ladder graph
    - One-edge graph
    - Path graph
    - Star graph
    - Maximal bipartite graph
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # Check for complete graph
    if nx.is_connected(G) and m == n * (n - 1) // 2:
        return True

    # Check for cyclic graph
    if nx.cycle_basis(G) and len(nx.cycle_basis(G)) == 1 and m == n:
        return True

    # Check for ladder graph
    if nx.is_connected(G) and m == 2 * (n - 1) and n % 2 == 0:
        return True

    # Check for one-edge graph
    if m == 1:
        return True

    # Check for path graph
    if nx.is_connected(G) and nx.is_tree(G) and len(list(nx.degree(G))) == 2:
        return True

    # Check for star graph
    if nx.is_tree(G) and sorted(dict(G.degree()).values())[-1] == n - 1:
        return True

    return False

def generate_valid_graph(vertices, edge_prob):
    """
    Generate a valid graph given the number of vertices and edge probability.
    Continuously tries until a valid graph is found.
    """
    attempt = 0
    while True:
        attempt += 1
        # Generate a random graph
        G = nx.erdos_renyi_graph(vertices, edge_prob)

        if not is_excluded_graph(G):  # Check if it's a valid graph
            matrix = nx.to_numpy_array(G, dtype=int)
            bitstring = adjacency_matrix_to_string(matrix)
            return bitstring, matrix