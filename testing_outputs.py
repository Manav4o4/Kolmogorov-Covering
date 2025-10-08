from kolmogorov_covering import kolmogorov_covering
from Generate_data_set.generate_erdos_renyi_graph import generate_erdos_renyi_graph
import matplotlib.pyplot as plt
from scripts.RNN_Graph_Classification import model
import os

model.eval()

def k_covering():

    vertex_list = [15, 20, 30, 40, 50, 60]

    for vertices in vertex_list:

        k_complexity = 0

        num_iterations = 0

        k_complexity_list = []
        
        prob_list = []

        iteration_list = []

        prob = 0.05

        prob_list.append(prob)

        for j in range(0,30,1):

            graph = generate_erdos_renyi_graph(vertices, prob)

            k_complexity_temp, num_iterations_temp = kolmogorov_covering(graph, model, False)

            k_complexity += k_complexity_temp

            num_iterations += num_iterations_temp

        k_complexity_list.append(k_complexity // 30)

        iteration_list.append(num_iterations // 30)

        for i in range(1, 11, 1):

            prob = i/10

            prob_list.append(prob)

            k_complexity = 0

            num_iterations = 0

            for j in range(0,30,1):

                graph = generate_erdos_renyi_graph(vertices, prob)

                k_complexity_temp, num_iterations_temp = kolmogorov_covering(graph, model, False)

                k_complexity += k_complexity_temp

                num_iterations += num_iterations_temp

            k_complexity_list.append(k_complexity // 30)

            iteration_list.append(num_iterations // 30)

        
        plt.figure(figsize=(10, 8))
        plt.plot(prob_list, k_complexity_list, label='K-Complexity', marker='o', color='blue')
        plt.xlabel('Probability')
        plt.ylabel('K-Complexity')
        plt.title('K-Complexity vs Probability')
        plt.legend()
        filename = f"k-complexity_vs_probability {vertices} vertices.png"
        save_path = os.path.join('K-Complexity Plots from 2log(n)+5', filename)
        plt.savefig(save_path)

        # New figure for iterations
        plt.figure(figsize=(10, 8))
        plt.plot(prob_list, iteration_list, label='Number of Iterations', marker='x', color='orange')
        plt.xlabel('Probability')
        plt.ylabel('Number of Iterations')
        plt.title('Number of Iterations vs Probability')
        plt.legend()
        filename = f"iteratations_vs_probability {vertices} vertices.png"
        save_path = os.path.join('K-Complexity Plots from 2log(n)+5', filename)
        plt.savefig(save_path)

k_covering()
