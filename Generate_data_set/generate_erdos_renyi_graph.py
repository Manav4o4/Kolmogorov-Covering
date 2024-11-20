import networkx as nx
import numpy as np

def generate_erdos_renyi_graph(vertices, edge_prob):
    # Generate an Erdős-Rényi random graph
    G = nx.erdos_renyi_graph(vertices, edge_prob)
    # Convert to adjacency matrix
    adjacency_matrix = nx.to_numpy_array(G, dtype=int)
    return adjacency_matrix



