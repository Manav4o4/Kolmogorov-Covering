def adjacency_matrix_to_edges(matrix):
    n = len(matrix)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):  # Only look at the upper triangle (j > i)
            if matrix[i][j] == 1:
                edges.append((i + 1, j + 1))  # Convert to 1-based indexing if needed
    return edges