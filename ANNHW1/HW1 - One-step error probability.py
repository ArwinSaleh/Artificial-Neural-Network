import random as rnd
import numpy as np

N = 120
n = 10 ** 5
patterns = [12, 24, 48, 70, 100, 120]
p_error = []


def main(boolean):
    for i in patterns:
        nr_errors = 0

        for j in range(0, n):  # 10^5 iterations for every number of patterns.
            p = generate_patterns(i)

            p_input = rnd.randint(0, i)
            m = rnd.randint(0, N - 1)

            w_matrix = (1 / N) * np.dot(p[m, :], np.conj(p).T)

            if boolean == 1:  # Choose task
                w_matrix[m] = 0  # This is the only real difference between the two tasks,
                # we set the weights to zero for the first task.

            s_0 = sgn(p[m, p_input - 1])
            s_1 = sgn(np.dot(w_matrix, p[:, p_input - 1]))

            if s_0 != s_1:
                nr_errors = nr_errors + 1

        p_temp = nr_errors / n
        p_error.append(p_temp)

        print("P_error = " + p_error.__str__())
    return p_error


def generate_patterns(p_n):
    pattern_matrix = np.random.randint(-1, 1, (N, p_n))
    pattern_matrix[pattern_matrix == 0] = 1

    # Here we generate a matrix of 1:s and -1:s.

    return pattern_matrix


def sgn(i):  # This method is used to define the signum function at signum(0).
    if i == 0:
        return 1
    else:
        return np.sign(i)


def task1a():
    print(main(1))


def task1b():
    print(main(0))

task1a()   # UNCOMMENT FOR RESULT FROM TASK 1A
#task1b()   # UNCOMMENT FOR RESULT FROM TASK 1B
