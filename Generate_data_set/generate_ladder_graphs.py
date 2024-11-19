def generate_ladder_graph(n):
    matrix = [[0] * (2 * n) for _ in range(2 * n)]
    for i in range(n - 1):
        matrix[i][i + 1] = 1
        matrix[i + 1][i] = 1
        matrix[n + i][n + i + 1] = 1
        matrix[n + i + 1][n + i] = 1
    for i in range(n):
        matrix[i][n + i] = 1
        matrix[n + i][i] = 1
    return matrix
