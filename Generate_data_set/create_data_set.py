import pandas as pd
import random
import numpy as np
import sys

sys.set_int_max_str_digits(0)

from generate_complete_graph import generate_complete_graph
from generate_cycle_graphs import generate_cycle_graph
from generate_ladder_graphs import generate_ladder_graph
from generate_one_edge_graphs import generate_one_edge_graph
from generate_path_graphs import generate_path_graph
from generate_star_graphs import generate_star_graph
from generate_maximal_bipartite_graphs import generate_maximal_bipartite_graph
from adjacency_matrix_encoding import adjacency_matrix_to_string
from generate_random_graphs import generate_valid_graph

data = []

for i in range(100):

    vertices = random.randint(1, 300)

    complete_adjacency = generate_complete_graph(vertices)
    edge_string = adjacency_matrix_to_string(complete_adjacency)
    data.append({"Edge Code": edge_string, "Label": 1})

    cycle_adjacency = generate_cycle_graph(vertices)
    edge_string = adjacency_matrix_to_string(cycle_adjacency)
    data.append({"Edge Code": edge_string, "Label": 1})

    ladder_adjacency = generate_ladder_graph(vertices)
    edge_string = adjacency_matrix_to_string(ladder_adjacency)    
    data.append({"Edge Code": edge_string, "Label": 1})

    one_edge_adjacency = generate_one_edge_graph(vertices)
    edge_string = adjacency_matrix_to_string(one_edge_adjacency)    
    data.append({"Edge Code": edge_string, "Label": 1})

    path_adjacency = generate_path_graph(vertices)
    edge_string = adjacency_matrix_to_string(path_adjacency)
    data.append({"Edge Code": edge_string, "Label": 1})

    star_adjacency = generate_star_graph(vertices)
    edge_string = adjacency_matrix_to_string(star_adjacency)
    data.append({"Edge Code": edge_string, "Label": 1})

    max_biaprtite_adjacenct = generate_maximal_bipartite_graph(vertices)
    edge_string = adjacency_matrix_to_string(max_biaprtite_adjacenct)
    data.append({"Edge Code": edge_string, "Label": 1})

    for j in range(10):

        prob = random.random()

        radnom_adjacency_string = generate_valid_graph(vertices, prob)
        data.append({"Edge Code": radnom_adjacency_string, "Label": 0})

df = pd.DataFrame(data, columns = ['Edge Code', 'Label'])

array_data = df.to_numpy()

np.save('data.npy', array_data)






