def generate_maximal_bipartite_graph(n):
    # Create a 2n x 2n matrix filled with 0's
    matrix = [[0] * (2 * n) for _ in range(2 * n)]
    
    # Connect every vertex in the first set (0 to n-1) with every vertex in the second set (n to 2n-1)
    for i in range(n):
        for j in range(n, 2 * n):
            matrix[i][j] = 1
            matrix[j][i] = 1
    
    return matrix
