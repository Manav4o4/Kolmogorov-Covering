def generate_star_graph(n):
    matrix = [[0] * n for _ in range(n)]
    for i in range(1, n):
        matrix[0][i] = 1
        matrix[i][0] = 1
    return matrix

