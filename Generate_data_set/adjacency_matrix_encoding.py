import sys
sys.set_int_max_str_digits(0)

def adjacency_matrix_to_string(matrix):
    result = []
    n = len(matrix)
    for j in range(n):  # Iterate over columns
        for i in range(j):  # Iterate over rows above the diagonal
            result.append(str(matrix[i][j]))
    return ''.join(result)


