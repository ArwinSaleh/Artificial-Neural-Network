import numpy as np
from numpy import genfromtxt

x_temp = genfromtxt('input_data_numeric.csv', delimiter=',')

index_u = x_temp[:, 0]  # Unused, these are the elements we DON'T use.

x_1 = x_temp[:, 1]

x_2 = x_temp[:, 2]

x_3 = x_temp[:, 3]

x_4 = x_temp[:, 4]

# We store the columns of 'input_data_numeric.csv' without including the 0:th column u = 1,..,16
x_data_temp = [x_1, x_2, x_3, x_4]

X_data = np.asmatrix(x_data_temp).transpose()

t_A = [1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1]
t_B = [-1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1]
t_C = [1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1]
t_D = [1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1]
t_E = [-1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1]
t_F = [1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1]

t_temp = [t_A, t_B, t_C, t_D, t_E, t_F]
T_data = np.array(t_temp)


class StochasticGradientDescent:

    def __init__(self):

        self.n = 0.02  # Learning rate
        self.W = initialise_weights()
        self.theta = generate_threshold()
        self.X = np.zeros((4, 1))
        self.O = np.zeros((16, 1))
        self.T = np.zeros((16, 1))
        self.delta = 0

    def update_error(self, u_index):
        error_temp = (self.T[u_index] - 0.5 * self.O[u_index] * self.g_prime())
        self.delta = error_temp.item()

    def train(self):
        self.update_threshold()
        self.update_weights()

    def update_threshold(self):
        self.theta = self.theta - self.n * self.delta

    def update_weights(self):
        self.W = self.W + (self.n * np.dot(self.delta, self.X).T)

    def generate_b(self):
        x_u = np.array(self.X)
        return -self.theta + np.dot(self.W.T, x_u.T)

    def generate_output(self):
        b_u = self.generate_b()
        return np.tanh(0.5 * b_u)

    def update_output(self, u_index):
        self.O[u_index] = self.generate_output()

    def g_prime(self):
        b_u = self.generate_b()
        return 1 - np.tanh(0.5 * b_u) ** 2

    def linearly_separable(self):

        output = self.O.copy()
        target = self.T.copy().T

        output[output == 0] = 1
        output = np.sign(output)
        output = np.round(output.transpose()).astype(int)

        if (output == target).all():
            return True
        else:
            return False


def initialise_weights():
    weights = np.random.uniform(-0.2, 0.2, (4, 1))
    return np.asmatrix(weights)


def generate_threshold():
    return np.random.uniform(-1, 1)


if __name__ == "__main__":

    structure = StochasticGradientDescent()
    maximum_iterations = 10 ** 4

    function_names = ['A', 'B', 'C', 'D', 'E', 'F']
    linearly_separable_list = []

    name_index = 0

    for t in T_data:  # We go through each input-pattern

        structure.T = np.asmatrix(t).copy()
        current_boolean_function = function_names[name_index]
        structure.T = t.copy()

        for i in range(0, 10):  # number of retries

            for j in range(0, maximum_iterations):

                u = np.random.randint(0, 16)

                structure.X = np.asarray(X_data[u, :])

                structure.update_output(u)
                structure.update_error(u)

                structure.train()

                if structure.linearly_separable():
                    print("\n" + "The following boolean function is linearly separable: " + current_boolean_function)
                    linearly_separable_list.append(current_boolean_function)
                    break

            if structure.linearly_separable():
                break
            else:
                print("\n" + "The following boolean function is NOT linearly separable: " + current_boolean_function)

            print("\n" + "Retry " + str(i + 1) + ":")

            structure.__init__()
            structure.T = t.copy()

        structure.__init__()

        name_index = name_index + 1

    print("\n" + "The following functions are linearly separable: " + str(linearly_separable_list))
