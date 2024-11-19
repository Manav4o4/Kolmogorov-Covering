def factors(n):
    result = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            result.append(i)
            if i != n // i:  # Avoid adding the square root twice for perfect squares
                result.append(n // i)
    return sorted(result)

def hashmap_shift_coef(edge_set, vertices):

    factor_list = factors(vertices)

    for current_factor in factor_list:

        edge_set_copy = edge_set.copy()

        for j in range(len(edge_set_copy)):

            left_vertex = 0

            right_vertex = 0

            if edge_set_copy[j][0] + current_factor > vertices:

                left_vertex = (edge_set_copy[j][0] + current_factor) - vertices

            else:

                left_vertex = edge_set_copy[j][0] + current_factor

            if edge_set_copy[j][1] + current_factor > vertices:

                right_vertex = (edge_set_copy[j][1] + current_factor) - vertices

            else:

                right_vertex = edge_set_copy[j][1] + current_factor

            edge_set_copy[j] = (left_vertex, right_vertex)

        for k in range(len(edge_set_copy)):

            if edge_set_copy[k] in edge_set or edge_set_copy[k][::-1] in edge_set:

                if k == len(edge_set) - 1:

                    return current_factor

            else:

                break 

U = [1, 2, 3, 4]  # Vertex class U
V = [5, 6, 7, 8, 9]  # Vertex class V

edges = [(u, v) for u in U for v in V]
                
test_list = [[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)], # Boundary Graph: O(1) K-Complexity
            [(1, 2)], # Graph with a single edge: O(1) K-Complexity
            [(1, 2), (3, 4), (5, 6)], # Ladder Graph: O(1) K-Complexity
            [(1, 4), (1, 5), (1, 6), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6)], # Maximal Bipartite Graph: Scales with n, for even |V|, shift coefficient = |V|/2
            edges # Maximal Biaprtite on odd vertices, scales with n. Shift Coefficient = |V|
            ]

vertices = [6, 6, 6, 6, 9]

for i in range(len(test_list)):

    # print(hashmap_shift_coef(test_list[i], vertices[i]))

    pass