from collections import deque
import numpy as np

def is_cyclic(adj_matrix):

    def dfs(node):
        visited.append(node)
        for neighbor, is_edge in enumerate(adj_matrix[node]):
            if is_edge and neighbor not in visited:
                dfs(neighbor)

    # Ensure the graph is valid
    num_nodes = len(adj_matrix)

    # Check if all vertices have degree 2
    for node in range(num_nodes):
        if sum(adj_matrix[node]) != 2:
            return False

    # Perform DFS
    visited = []
    dfs(0)  # Start from the first vertex

    if len(visited) == len(adj_matrix):
        print("Cyclic")
        return True
    
    print("Hi")
    return False


def is_bipartite_and_get_classes(adj_matrix):
    n = len(adj_matrix)
    colors = [-1] * n  # -1 means uncolored, 0 and 1 are the two colors
    sets = {0: set(), 1: set()}

    for start in range(n):  # Handle disconnected components
        if colors[start] == -1:
            queue = deque([start])
            colors[start] = 0
            sets[0].add(start)

            while queue:
                node = queue.popleft()
                current_color = colors[node]
                next_color = 1 - current_color

                for neighbor in range(n):
                    if adj_matrix[node][neighbor] == 1:  # There is an edge
                        if colors[neighbor] == -1:  # Not yet colored
                            colors[neighbor] = next_color
                            sets[next_color].add(neighbor)
                            queue.append(neighbor)
                        elif colors[neighbor] == current_color:  # Conflict found
                            return False, None, None

    return True, sets[0], sets[1]

def check_maximal_bipartite(adj_matrix, set_u, set_v):
    for u in set_u:
        for v in set_v:
            if adj_matrix[u][v] == 0:  # Missing edge between sets
                return False
    return True

def is_maximal_bipartite(adj_matrix):
    is_bipartite, set_u, set_v = is_bipartite_and_get_classes(adj_matrix)

    if not is_bipartite:
        return "The graph is not bipartite.", None, None

    if check_maximal_bipartite(adj_matrix, set_u, set_v) == True:
        print("MB")
        return True
    else:
        return False
    
def is_complete(adj_matrix):
    n = len(adj_matrix)
    for i in range(n):
        for j in range(i+1, n):
            if adj_matrix[i][j] == 0:
                return False
    print("C")
    return True


# Function to check if a graph is a path graph
def is_path(graph) -> bool:
    """
    Checks if the graph is a path graph.
    :param graph: Adjacency matrix of the undirected graph.
    :return: True if the graph is a path, False otherwise.
    """
    degree = np.sum(graph, axis=1)
    
    degree_one_count = np.count_nonzero(degree == 1)
    degree_two_count = np.count_nonzero(degree == 2)

    if degree_one_count == 2 and degree_two_count == len(graph) - 2:
        print("P")
        return True
    
    return False

