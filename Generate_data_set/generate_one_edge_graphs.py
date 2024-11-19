def generate_one_edge_graph(n):
    matrix = [[0] * n for _ in range(n)]
    if n > 1:
        matrix[0][1] = 1
        matrix[1][0] = 1
    return matrix
