from kolmogorov_covering import kolmogorov_covering
from Generate_data_set.generate_erdos_renyi_graph import generate_erdos_renyi_graph
from k_covering_helpers import string_to_tensor
from Generate_data_set.adjacency_matrix_encoding import adjacency_matrix_to_string
import matplotlib.pyplot as plt
from scripts.RNN_Graph_Classification import model
import torch
import os

model.eval()

def testing():
    # Vertex count to test
    vertex_list = [i for i in range(10, 200, 20)]

    # Separate probabilities into two ranges
    probability_ranges = {
        "0.0 - 0.5": [i / 10 for i in range(1, 6)],  # Probabilities: 0.1 to 0.5
        "0.5 - 1.0": [i / 10 for i in range(6, 11)]  # Probabilities: 0.6 to 1.0
    }

    for range_label, probabilities in probability_ranges.items():
        plt.figure(figsize=(12, 10))

        for prob in probabilities:
            averaged_outputs = []  # To store averaged outputs for each vertex count
            
            for vertices in vertex_list:
                output_sum = 0.0  # To sum outputs where output > 0.95
                num_samples = 500  # Number of graphs to generate
                
                for _ in range(num_samples):
                    # Generate a random graph
                    graph = generate_erdos_renyi_graph(vertices, prob)

                    # Convert adjacency matrix to string
                    graph_bitstring = adjacency_matrix_to_string(graph)

                    # Get model output
                    with torch.no_grad():
                        output = model(string_to_tensor(graph_bitstring))
                    
                    # Count outputs > 0.95
                    if output > 0.95:
                        output_sum += 1
                
                # Compute average of outputs > 0.95
                average_output = output_sum / num_samples
                averaged_outputs.append(average_output)
            
            # Add to plot for this probability
            plt.plot(vertex_list, averaged_outputs, label=f'Probability = {prob}', marker='o')
        
        # Finalize the plot for this range
        plt.xlabel('Vertices')
        plt.ylabel('Fraction of Outputs > 0.95')
        plt.title(f'Averaged Outputs > 0.95 vs. Vertices ({range_label})')
        plt.legend()
        
        # Save plot
        os.makedirs('Averaged Graph Plots', exist_ok=True)
        filename = f"averaged_output_{range_label.replace(' ', '_')}.png"
        save_path = os.path.join('Averaged Graph Plots', filename)
        plt.savefig(save_path)
        plt.close()

#testing()

def testing2():

    # Vertex count to test
    vertex_list = [i for i in range(10, 200, 20)]

    # Test for probabilities from 0.1 to 1.0
    for i in range(1, 11):
        prob = i / 10  # Convert to probability
        
        averaged_outputs = []  # To store averaged outputs for each vertex count
        
        for vertices in vertex_list:
            output_sum = 0.0  # To sum outputs where output > 0.95
            num_samples = 500  # Number of graphs to generate
            
            for _ in range(num_samples):
                # Generate a random graph
                graph = generate_erdos_renyi_graph(vertices, prob)

                # Convert adjacency matrix to string
                graph_bitstring = adjacency_matrix_to_string(graph)

                # Get model output
                with torch.no_grad():
                    output = model(string_to_tensor(graph_bitstring))
                
                # Count outputs > 0.95
                if output > 0.95:
                    output_sum += 1
            
            # Compute average of outputs > 0.95
            average_output = output_sum / num_samples
            averaged_outputs.append(average_output)
        
        # Plot results for this probability
        plt.figure(figsize=(12, 10))
        plt.plot(vertex_list, averaged_outputs, label=f'Probability = {prob}', marker='o', color='blue')
        plt.xlabel('Number of Vertices')
        plt.ylabel('Averaged Output Value (> 0.95)')
        plt.title(f'Averaged Output vs. Vertices (Probability = {prob})')
        plt.legend()
        
        # Save plot
        os.makedirs('Probability Graphs', exist_ok=True)
        filename = f"output_vs_vertices_prob_{prob}.png"
        save_path = os.path.join('Probability Graphs', filename)
        plt.savefig(save_path)
        plt.close()

# testing2()

def testing3():

    # Vertex count to test
    vertex_list = [i for i in range(10, 200, 20)]

    # Create a single figure
    plt.figure(figsize=(14, 10))

    # Test for probabilities from 0.1 to 1.0
    for i in range(1, 11):
        prob = i / 10  # Convert to probability
        
        averaged_outputs = []  # To store averaged outputs for each vertex count
        
        for vertices in vertex_list:
            output_sum = 0.0  # To sum outputs where output > 0.95
            num_samples = 500  # Number of graphs to generate
            
            for _ in range(num_samples):
                # Generate a random graph
                graph = generate_erdos_renyi_graph(vertices, prob)

                # Convert adjacency matrix to string
                graph_bitstring = adjacency_matrix_to_string(graph)

                # Get model output
                with torch.no_grad():
                    output = model(string_to_tensor(graph_bitstring))
                
                # Count outputs > 0.95
                if output > 0.95:
                    output_sum += 1
            
            # Compute average of outputs > 0.95
            average_output = output_sum / num_samples
            averaged_outputs.append(average_output)
        
        # Plot the results for this probability as a line on the graph
        plt.plot(vertex_list, averaged_outputs, label=f'Probability = {prob}', marker='o')

    # Finalize the graph
    plt.xlabel('Number of Vertices')
    plt.ylabel('Averaged Output Value (> 0.95)')
    plt.title('Averaged Output vs. Vertices for Different Probabilities')
    plt.legend(title="Probabilities")
    plt.grid(True)

    # Save the combined plot
    os.makedirs('Probability Graphs', exist_ok=True)
    filename = "output_vs_vertices_all_probabilities_single_graph.png"
    save_path = os.path.join('Probability Graphs', filename)
    plt.savefig(save_path)
    plt.show()  # Display the figure

testing3()

