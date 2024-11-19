import torch
import numpy as np

def matrix_to_string(matrix):

    result_dict = {}

    for i in range(len(matrix)):
        # Join each row (list) into a string and assign it to the dictionary
        result_dict[i + 1] = "".join(map(str, matrix[i]))

    return result_dict

def string_to_tensor(str):

    sequence = torch.tensor([int(bit) for bit in str], dtype=torch.float32)

    return sequence.view(1, -1, 1)

def delete_vertex(matrix, vertex):

    matrix = np.delete(matrix, vertex, axis=0)  # Delete the row
    matrix = np.delete(matrix, vertex, axis=1)  # Delete the column
    return matrix

def create_bitstring(subgraph, graph_dict):

    subgraph_bitstring = ''

    for vertex in subgraph:

        subgraph_bitstring = subgraph_bitstring + graph_dict[vertex][vertex:] # Builds the substring of the current subgraph

    return subgraph_bitstring

