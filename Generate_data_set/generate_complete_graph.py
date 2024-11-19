def generate_complete_graph(n):
    # Initialize an n x n matrix with 1s
    matrix = [[1 if i != j else 0 for j in range(n)] for i in range(n)]
    return matrix
