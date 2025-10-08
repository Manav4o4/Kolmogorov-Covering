import itertools
import math
import copy
from k_covering_helpers import create_subgraph_matrix, display_subgraph
from Generate_data_set.graph_checks import is_path, is_cyclic, is_complete, is_maximal_bipartite
from Generate_data_set.generate_erdos_renyi_graph import generate_erdos_renyi_graph

def updated_covering(graph):

    size = 2*math.ceil(math.log(len(graph), 2)) + 5

    vertex_list = [i for i in range(1, len(graph) + 1)]

    subsets_iterator = itertools.combinations(vertex_list, size)

    for subset in subsets_iterator:
        subgraph_matrix = copy.deepcopy(graph)  # Create a copy of the graph
        subgraph_matrix = create_subgraph_matrix(graph, subset, subgraph_matrix)  # Create subgraph matrix

        # Check conditions
        if not is_path(graph) and not is_cyclic(graph) and not is_complete(graph) and not is_maximal_bipartite(graph):
            count += 1  # Increment the counter if the subset meets the condition

    return count



    
graph = generate_erdos_renyi_graph(40, 0.05)
print(updated_covering(graph))
    



