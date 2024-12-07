import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

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

def display_graph(matrix_dict, count, vertices):

    matrix_copy_as_list = [list(map(int, row)) for row in matrix_dict.values()]

    G = nx.from_numpy_array(np.array(matrix_copy_as_list))

    labels = {i: key for i, key in enumerate(matrix_dict.keys())}  # Map index to keys

    pos = nx.spring_layout(G, k=5, scale=4, iterations=50)  # Position the nodes with a layout for better visualization

    filename = f"{count}th graph on {vertices} vertices.png"

    save_path = os.path.join('Graph_outputs', filename)

    plt.clf()
    nx.draw(G, pos, labels=labels, with_labels=True, node_color="lightblue", font_weight="bold")
    plt.savefig(save_path)

def create_subgraph_matrix(graph, subgraph, subgraph_matrix):

    for vertex in range(len(graph), 0, -1):
                    
        if vertex not in subgraph:

            numpy_graph = np.array(subgraph_matrix)

            subgraph_matrix = delete_vertex(numpy_graph, vertex-1).tolist() # -1 for the indices

    return subgraph_matrix

def delete_subgraph(subgraph, graph, vertices):
    
    for vertex in range(len(subgraph)-1, -1, -1): # Delete vertices from the basic subgraph's subset

        numpy_graph = np.array(graph)

        print("deleting ", subgraph[vertex], "from index", subgraph[vertex]-1)

        graph = delete_vertex(numpy_graph, subgraph[vertex]-1).tolist() # -1 for the indices

        vertices -= 1
        
    return graph, vertices

def vis_dict_update(subgraph, vertex_dict, graph_dict):

    temp_dict = {}

    for key in vertex_dict.keys():

        if key not in subgraph:

            temp_dict[key] = vertex_dict[key]

    vertex_dict = temp_dict

    sorted_items = sorted(vertex_dict.items())  # Sort the items by keys

    vertex_dict = {new_key: value for new_key, (_, value) in enumerate(sorted_items, start=1)}  # Renumber keys

    print(vertex_dict)

    vertex_list = list(vertex_dict.values())

    if len(vertex_list) == 1:

        graph_dict[vertex_list[0]] = graph_dict[list(graph_dict.keys())[0]]

    else:

        temp_dict = {}

        dict_keys = list(graph_dict.keys())

        for i in range(len(vertex_list)):

            temp_dict[vertex_list[i]] = graph_dict[dict_keys[i]]

    return vertex_dict, temp_dict
