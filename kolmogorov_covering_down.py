import math, torch
from itertools import combinations
from scripts.RNN import model
from Generate_data_set.adjacency_matrix_encoding import adjacency_matrix_to_string
import copy
from k_covering_helpers import matrix_to_dict, string_to_tensor, display_graph, create_subgraph_matrix, delete_subgraph, vis_dict_update

def kolmogorov_covering_down(graph, model, flag):

    '''
        The dictionary/hash map are for visualizing the graph, I do not consider related opertaions in the asymptotic analysis.
    '''

    count = 0

    vertices = len(graph)

    vertices_dynamic = len(graph) # The number of vertices that changes when we delete a subgraph

    complexity = math.comb(vertices, 2)

    subgraphs_iterated_over = 0

    graph_dict = matrix_to_dict(graph)

    vertex_dict = {i + 1 : i + 1 for i in range(vertices)} # For visualization, not needed for the computation

    if (flag == True):

        display_graph(graph_dict, count, vertices)

    lower_bound = math.ceil(2*math.log(vertices, 2)) + 5

    upper_bound = vertices_dynamic

    while upper_bound >= lower_bound:  

        print(upper_bound)

        if len(graph) < math.ceil(2*math.log(vertices_dynamic, 2)) + 5: # If |graph| < |subgraph|, then break

            break

        graph_dict = matrix_to_dict(graph)

        subsets = iter(combinations([j for j in range(1, vertices_dynamic + 1)], upper_bound)) # Gives us all possible subseuquences of the lebelling of size 'i'

        subsets_iterator = iter(subsets)

        while True:

            try:

                subgraph = next(subsets_iterator)  # Get the next subgraph

            except StopIteration:

                break  # Exit the loop when there are no more subsets

            placeholder = True

            if placeholder:

                subgraphs_iterated_over += 1

                subgraph = list(subgraph)

                subgraph_matrix = copy.deepcopy(graph)  # Create a copy of the graph which will be modified for the subgraph

                subgraph_matrix = create_subgraph_matrix(graph, subgraph, subgraph_matrix)  # Create the matrix of the subgraph after deleting vertices that are not in the subset

                subgraph_bitstring = adjacency_matrix_to_string(subgraph_matrix)  # Creates the bitstring for the current subgraph with n choose 2 bits

                with torch.no_grad():
                    output = model(string_to_tensor(subgraph_bitstring))  # Evaluates the substring, which is converted to a tensor on the model

                if output.item() > 0.95: # If the graph is basic with probability 95%, then update the complexity, graph_dict

                    graph, vertices_dynamic = delete_subgraph(subgraph, graph, vertices_dynamic)  # Deletes the subgraph

                    graph_dict = matrix_to_dict(graph)  # Update the graph dictionary

                    complexity -= math.comb(len(subgraph), 2)  # Update the complexity

                    vertex_dict, graph_dict = vis_dict_update(subgraph, vertex_dict, graph_dict)  # Updates the vertex list and graph_dict for visualization

                    if flag:
                         
                        count += 1

                        display_graph(graph_dict, count, vertices)

                    if len(graph) < math.ceil(2 * math.log(vertices_dynamic, 2)) + 5:  # If |graph| < |size of required subgraph|, then break

                        break

                    subsets_iterator = iter(combinations([j for j in range(1, vertices_dynamic + 1)], upper_bound))

                    lower_bound = math.ceil(2*math.log(vertices_dynamic, 2)) + 5

                    upper_bound = vertices_dynamic

        upper_bound -= 1

    return complexity, subgraphs_iterated_over
