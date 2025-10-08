import pandas as pd
import random
import numpy as np
from generate_complete_graph import generate_complete_graph
from generate_cycle_graphs import generate_cycle_graph
from generate_path_graphs import generate_path_graph
from generate_maximal_bipartite_graphs import generate_maximal_bipartite_graph
from adjacency_matrix_encoding import adjacency_matrix_to_string
from generate_random_graphs import generate_valid_graph

# Initialize data list
data = []

# Generate data for basic graphs
for i in range(20):
    vertices = random.randint(1, 100)
    
    # Complete graph
    complete_adjacency = generate_complete_graph(vertices)
    edge_string = adjacency_matrix_to_string(complete_adjacency)
    data.append({"Edge Code": edge_string, "Type": "complete", "Label": 1})
    
    # Cycle graph
    cycle_adjacency = generate_cycle_graph(vertices)
    edge_string = adjacency_matrix_to_string(cycle_adjacency)
    data.append({"Edge Code": edge_string, "Type": "cycle", "Label": 1})
    
    # Path graph
    path_adjacency = generate_path_graph(vertices)
    edge_string = adjacency_matrix_to_string(path_adjacency)
    data.append({"Edge Code": edge_string, "Type": "path", "Label": 1})
        
    # Maximal bipartite graph
    max_bipartite_adjacency = generate_maximal_bipartite_graph(vertices)
    edge_string = adjacency_matrix_to_string(max_bipartite_adjacency)
    data.append({"Edge Code": edge_string, "Type": "maximal_bipartite", "Label": 1})
    
    # Random non-basic graphs
    for j in range(4):
        prob = random.random()
        random_adjacency_string, _ = generate_valid_graph(vertices, prob)
        data.append({"Edge Code": random_adjacency_string, "Type": "random", "Label": 0})

# Create a DataFrame

df = pd.DataFrame(data, columns=["Edge Code", "Type", "Label"])

# Convert the DataFrame to a numpy array and save
array_data = df.to_numpy()
np.save("data.npy", array_data)
