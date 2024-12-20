def vertex_shift_coefficients(matrix):

    matrix_dict = matrix_to_dict(matrix)

    vertices = len(matrix_dict)

    factor_list = factors(vertices)

    for current_factor in factor_list:

        matrix_dict_copy = {k: list(v) for k, v in matrix_dict.items()}

        for i in range(1, vertices + 1):
            
            matrix_dict_copy[i] = shift_list(matrix_dict[i], current_factor)

        for row in range(1, vertices):

            if row == len(matrix_dict_copy) - 1 and matrix_dict_copy[row] in matrix_dict.values():

                return current_factor
            
            elif matrix_dict_copy[row] in matrix_dict.values():

                continue

            else:

                break

def matrix_to_dict(matrix):

    dict = {}

    for i in range(len(matrix)):

        dict[i + 1] = matrix[i]

    return dict

def shift_list(lst, shift):
    # Handle the case where the list is empty
    if not lst:
        return lst

    # Get the length of the list
    n = len(lst)

    shift = shift % n  # This makes sure that shifts larger than the list length wrap around

    # Perform the shift
    return lst[-shift:] + lst[:-shift]


def factors(n):
    result = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            result.append(i)
            if i != n // i:  # Avoid adding the square root twice for perfect squares
                result.append(n // i)
    return sorted(result)


matrix = [  # Vertex labels (header row)
    [0, 0, 0, 1, 1, 1],  # Vertex 1 connections to vertices in Set B
    [0, 0, 0, 1, 1, 1],  # Vertex 2 connections to vertices in Set B
    [0, 0, 0, 1, 1, 1],  # Vertex 3 connections to vertices in Set B
    [1, 1, 1, 0, 0, 0],  # Vertex 4 connections to vertices in Set A
    [1, 1, 1, 0, 0, 0],  # Vertex 5 connections to vertices in Set A
    [1, 1, 1, 0, 0, 0]   # Vertex 6 connections to vertices in Set A
]

print(vertex_shift_coefficients(matrix))
