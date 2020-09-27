import numpy as np
import random
import math
import time

class LinearSeparability:

    def __init__(self):
        self.weights = np.random.uniform(-0.2, 0.2, (4, 1))
        self.threshold = random.uniform(-1, 1)
        self.learning_rate = 0.02
        file_name: str = "input_data_numeric.csv"
        # noinspection PyTypeChecker
        self.possible_inputs = np.loadtxt(open(file_name, "r"), delimiter=",", usecols=(1, 2, 3, 4))
        self.input = np.zeros((4,1))
        self.output = np.zeros((16, 1))
        A = [1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1]
        B = [-1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1]
        C = [1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1]
        D = [1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1]
        E = [-1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1]
        F = [1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1]
        self.boolean_functions_name = ['C', 'E', 'F', 'D', 'A', 'B']
        self.boolean_functions = np.asmatrix([C, E, F, D, A, B]).transpose()
        self.target = []
        self.beta = 1/2

        self.error = 0

    def train_network(self):
        self.weights += (self.learning_rate * np.asmatrix(self.error * self.input)).transpose()
        self.threshold -= self.learning_rate * self.error

    def compute_error(self, mu):
        g_prime = self.compute_g_prime()
        error = (self.target[mu] - self.output[mu]) * g_prime * self.beta
        self.error = error.item()

    def compute_g_prime(self):
        b = self.compute_b()
        g_prime = (1 - math.tanh(self.beta*b)**2)
        return g_prime

    def compute_b(self):
        pattern = np.asmatrix(self.input)
        b = -self.threshold + np.dot(self.weights.transpose(), pattern.transpose())
        return b

    def compute_g(self):
        b = self.compute_b()
        g = math.tanh(self.beta*b)
        return g

    def update_output(self, mu):
        self.output[mu] = self.compute_g()

    def check_linear_separability(self):
        temp_output = self.output.copy()
        temp_output[temp_output >= 0] = 1
        temp_output[temp_output < 0] = -1

        if np.array_equal(temp_output, self.target):
            return True
        else:
            return False


network = LinearSeparability()
iterations = 10**4
tries = 10
input_neurons = 4
linearly_separable_functions = []

for m in range(network.boolean_functions.shape[1]):
    linearly_separable = False
    network.target = np.asmatrix(network.boolean_functions[:, m]).copy()

    for k in range(tries):
        for j in range(iterations):

            for mu in range(16):
                network.input = np.asarray(network.possible_inputs[mu, :])
                network.update_output(mu)

                network.compute_error(mu)
                network.train_network()

            linearly_separable = network.check_linear_separability()

            if linearly_separable:
                print("Boolean function " + network.boolean_functions_name[m] + " is linearly separable!")
                linearly_separable_functions.append(network.boolean_functions_name[m])
                break

        if linearly_separable:
            break
        else:
            print("Boolean function " + network.boolean_functions_name[m] + " is not linearly separable!")

        print("Try number: " + str(k + 1))
        network.__init__()
        network.target = np.asmatrix(network.boolean_functions[:, m]).copy()

    network.__init__()

print("These functions are linearly separable: " + ', '.join(linearly_separable_functions))