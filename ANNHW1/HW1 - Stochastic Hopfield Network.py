import numpy as np
import random as rnd

N = 200
T = 2 * 10 ** 5
beta = 2
reps = 100


def sigmoid(b):
    z = 1 / (1 + np.exp(-2 * beta * b))
    return z


def random_number_dist(nr1, nr2, prob1, prob2):
    numbers = [nr1, nr2]
    distribution = [prob1, prob2]
    rand_dist = rnd.choices(numbers, distribution)
    return rand_dist[0]


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


def main(p_matrix, w_matrix):
    m_u = []  # Order parameter
    for i in range(0, reps):

        last_state = p_matrix[:, 1]
        iterations = np.zeros(reps)

        for j in range(0, T):
            next_state = last_state

            m = rnd.randint(0, N)  # Choose neuron randomly
            b_m = np.dot(w_matrix[m, :], last_state)
            p_b = sigmoid(b_m)

            next_state[m] = random_number_dist(1, -1, p_b, 1 - p_b)
            iterations[j] = (1 / N) * p_matrix[:, 1] * np.transpose(next_state)
            last_state = next_state

        m_u[i] = (1 / T) * sum(iterations)

    m_u_avg = (1 / reps) * sum(m_u)

    return m_u_avg


def task1():
    p = 7
    p_matrix = generate_patterns(p)
    w_matrix = generate_weight(p, p_matrix)
    res = main(p_matrix, w_matrix)
    print(res)


def task2():
    p = 45
    p_matrix = generate_patterns(p)
    w_matrix = generate_weight(p, p_matrix)
    res = main(p_matrix, w_matrix)

task1()
# task2()
