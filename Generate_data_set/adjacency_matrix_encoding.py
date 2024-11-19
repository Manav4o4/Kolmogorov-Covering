import sys
sys.set_int_max_str_digits(0)

def adjacency_matrix_to_string(matrix):
    # Step 1: Initialize an n x n adjacency matrix with 0s    
    # Step 3: Extract the upper triangular part (excluding the diagonal)
    result = []
    for j in range(len(matrix)):
        for i in range(j + 1, len(matrix)):
            result.append(str(matrix[j][i]))
    
    # Step 4: Join the list into a single string
    return (''.join(result))


