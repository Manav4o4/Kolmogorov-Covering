from kolmogorov_covering import kolmogorov_covering
from Generate_data_set.generate_random_graphs import generate_erdos_renyi_graph
import math
import matplotlib.pyplot as plt
from RNN import model
import os

model.eval()

def k_covering_fifty():

    vertices = 30
    k_complexity_list = []
    prob_list = []
    iteration_list = []
    original_complexity_list = []

    for i in range(1, 11, 1):

        prob = i/10

        prob_list.append(prob)

        original_complexity = math.comb(vertices, 2)

        k_complexity = 0

        num_iterations = 0

        for j in range(0,10,1):

            graph = generate_erdos_renyi_graph(vertices, prob)

            k_complexity_temp, num_iterations_temp = kolmogorov_covering(graph, model)

            k_complexity += k_complexity_temp

            num_iterations += num_iterations_temp

        k_complexity_list.append(k_complexity // 10)

        iteration_list.append(num_iterations // 10)

        original_complexity_list.append(original_complexity)

    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_list, k_complexity_list, label='K-Complexity', marker='o', color='blue')
    plt.plot(prob_list, original_complexity_list, label='Starting Complexity', marker='p', color='red')
    plt.xlabel('Probability')
    plt.ylabel('Values')
    plt.title('K-Complexity and Starting Complexity vs Probability')
    plt.legend()
    save_path = os.path.join('Outputs', "k_complexity_vs_probability_30.png")
    plt.savefig(save_path)
    plt.grid(True)
    plt.show()

    # New figure for iterations
    plt.figure(figsize=(8, 6))
    plt.plot(prob_list, iteration_list, label='Number of Iterations', marker='x', color='orange')
    plt.xlabel('Probability')
    plt.ylabel('Number of Iterations')
    plt.title('Number of Iterations vs Probability')
    plt.legend()
    save_path = os.path.join('Outputs', "iteratations_vs_probability_30.png")
    plt.savefig(save_path)
    plt.grid(True)
    plt.show()

k_covering_fifty()