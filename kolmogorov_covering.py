import math, torch
import numpy as np
from itertools import combinations

def kolmogorov_covering(graph, model):

    vertices = len(graph)

    complexity = math.comb(vertices, 2)

    subgraphs_iterated_over = 0

    for i in range(vertices, math.ceil(2*math.log(vertices, 2) + 5), -1):

        if len(graph) < math.ceil(2*math.log(vertices, 2) + 5): # If |graph| < |subgraph|, then break

            break

        graph_dict = matrix_to_dict(graph)

        subsets = list(combinations(list(graph_dict.keys()), i)) # Gives us all possible subseuquences of the lebelling of size 'i'

        print(len(subsets))

        subgraphs_iterated_over += len(subsets)

        for subgraph in subsets:

            subgraph_bitstring = create_bitstring(subgraph, graph_dict) # Creates the bitstring for the current subgraph

            with torch.no_grad():
                output = model(string_to_tensor(subgraph_bitstring))

            # print(output.item(), i)

            if output.item() > 0.5: # If the graph is basic, then update the complexity, graph_dict

                for vertex in range(len(subgraph)-1, 0, -1):

                    numpy_graph = np.array(graph)

                    graph = delete_vertex(numpy_graph, subgraph[vertex-1]).tolist()

                graph_dict = matrix_to_dict(graph)

                complexity -= math.comb(len(subgraph), 2)

                if len(graph) < math.ceil(2*math.log(vertices, 2) + 5): # If |graph| < |subgraph|, then break

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

def create_bitstring(subgraph, graph_dict):

    subgraph_bitstring = ''

    for vertex in subgraph:

        subgraph_bitstring = subgraph_bitstring + graph_dict[vertex][vertex:] # Builds the substring of the current subgraph

    return subgraph_bitstring
            


    