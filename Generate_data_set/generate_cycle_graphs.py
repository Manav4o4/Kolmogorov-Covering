def generate_cycle_graph(n):

    if n <= 1:  
        return [[0] * n for _ in range(n)] 

    matrix = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        matrix[i][i + 1] = 1
        matrix[i + 1][i] = 1
    matrix[0][n - 1] = 1
    matrix[n - 1][0] = 1
    return matrix
