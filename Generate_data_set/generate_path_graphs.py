def generate_path_graph(n):
    matrix = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        matrix[i][i + 1] = 1
        matrix[i + 1][i] = 1
    return matrix
