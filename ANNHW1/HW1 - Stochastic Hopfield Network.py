import numpy as np

N = 200
T = 2 * 10 ** 5
p = 7


def generate_patterns(p):
    p_matrix = np.random.randint(-1, 1, (N, p))
    p_matrix[p_matrix == 0] = 1

    # Generating matrix a matrix of 1:s and -1:s

    return p_matrix


