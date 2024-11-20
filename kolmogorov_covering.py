import math, torch
import numpy as np
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
from Generate_data_set.generate_erdos_renyi_graph import generate_erdos_renyi_graph
from Generate_data_set.generate_random_graphs import generate_valid_graph
from scripts.RNN import model
from Generate_data_set.adjacency_matrix_encoding import adjacency_matrix_to_string
import os
import copy

def kolmogorov_covering(graph, model, flag):

    count = 0

    vertices = len(graph)

    complexity = math.comb(vertices, 2)

    subgraphs_iterated_over = 0

    graph_dict = matrix_to_dict(graph)

    if (flag == True):

        display_graph(graph_dict, count)

    for i in range(vertices, math.ceil(2*math.log(vertices, 2) + 5), -1):

        if len(graph) < math.ceil(2*math.log(vertices, 2) + 5): # If |graph| < |subgraph|, then break

            break

        graph_dict = matrix_to_dict(graph)

        subsets = list(combinations(list(graph_dict.keys()), i)) # Gives us all possible subseuquences of the lebelling of size 'i'

        print(len(subsets))

        subgraphs_iterated_over += len(subsets)

        for subgraph in subsets:

            subgraph_matrix = copy.deepcopy(graph) # Create a copy of the graph which will be modified for the subgraph

            subgraph_matrix = create_subgraph_matrix(graph, subgraph, subgraph_matrix) # Create the maxtrix of the subgraph after deleting vertices that are not in the subset

            subgraph_bitstring = adjacency_matrix_to_string(subgraph_matrix) # Creates the bitstring for the current subgraph with n choose 2 bits

            with torch.no_grad():
                output = model(string_to_tensor(subgraph_bitstring))

            if output.item() > 0.5: # If the graph is basic, then update the complexity, graph_dict

                for vertex in range(len(subgraph)-1, 0, -1):

                    numpy_graph = np.array(graph)

                    graph = delete_vertex(numpy_graph, subgraph[vertex-1]).tolist() # -1 for the indices

                graph_dict = matrix_to_dict(graph)

                complexity -= math.comb(len(subgraph), 2)

                if (flag == True):

                    count += 1

                    display_graph(graph_dict, count)

                if len(graph) < math.ceil(2*math.log(vertices, 2) + 5): # If |graph| < |size of required subgraph|, then break

                    break

    return complexity, subgraphs_iterated_over

def matrix_to_dict(graph):

    result_dict = {}

    for i in range(len(graph)):
        # Join each row (list) into a string and assign it to the dictionary
        result_dict[i + 1] = "".join(map(str, graph[i]))

    return result_dict

def string_to_tensor(str):

    sequence = torch.tensor([int(bit) for bit in str], dtype=torch.float32)

    return sequence.view(1, -1, 1)

def delete_vertex(graph, vertex):

    graph = np.delete(graph, vertex, axis=0)  # Delete the row
    graph = np.delete(graph, vertex, axis=1)  # Delete the column
    return graph

def display_graph(matrix_dict, count):

    matrix_copy_as_list = [list(map(int, row)) for row in matrix_dict.values()]

    print(matrix_dict)

    G = nx.from_numpy_array(np.array(matrix_copy_as_list))

    labels = {i: f"{i+1}" for i in range(len(matrix_copy_as_list))}  # Using v1, v2, ... as labels
    pos = nx.spring_layout(G)  # Position the nodes with a layout for better visualization

    print(G)

    filename = f"{count}th graph on 30 vertices.png"

    save_path = os.path.join('Graph_outputs', filename)

    nx.draw(G, pos, labels=labels, with_labels=True, node_color="lightblue", font_weight="bold")
    plt.savefig(save_path)
    plt.grid(True)
    plt.show()
    plt.clf()

def create_subgraph_matrix(graph, subgraph, subgraph_matrix):

    for vertex in range(len(graph), 0, -1):
                    
        if vertex not in subgraph:

            numpy_graph = np.array(subgraph_matrix)

            subgraph_matrix = delete_vertex(numpy_graph, vertex-1).tolist() # -1 for the indices

    return subgraph_matrix
