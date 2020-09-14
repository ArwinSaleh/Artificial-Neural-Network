import numpy as np

N = 200
T = 2 * 10 ** 5
beta = 2


def generate_patterns(p):
    p_matrix = np.random.randint(-1, 1, (N, p))
    p_matrix[p_matrix == 0] = 1

    # Generating matrix a matrix of 1:s and -1:s

    return p_matrix


def generate_weight(p, patterns):
    w_matrix = np.zeros((N, N))

    for i in range(p):
        w_matrix = w_matrix + np.dot(np.transpose(np.array([patterns[:, i]])), np.array([patterns[:, i]]))

    w_matrix = w_matrix / N
    np.fill_diagonal(w_matrix, 0)

    return w_matrix


def task1():
    p = 7
    p_matrix = generate_patterns(p)

    w_matrix = generate_weight(p, p_matrix)
    print(w_matrix)


task1()
