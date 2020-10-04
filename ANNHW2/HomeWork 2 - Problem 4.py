import numpy as np
from numpy import genfromtxt
import os

training_set = genfromtxt('training_set.csv', delimiter=',')
validation_set = genfromtxt('validation_set.csv', delimiter=',')

x1_u = training_set[:, 0]
x2_u = training_set[:, 1]
t_u = training_set[:, 2]

x1_u_val = validation_set[:, 0]
x2_u_val = validation_set[:, 1]
t_u_val = validation_set[:, 2]

u_index = len(t_u)
p_val = len(t_u_val)

maximum_iterations = 10 ** 3
epochs = 10 ** 3
layers = 3

length_training = len(training_set)
length_validation = len(validation_set)


class TwoLayerPerceptron:

    def __init__(self):
        self.n = 0.02  # Learning rate
        self.X = np.array([x1_u, x2_u]).T  # Inputs
        self.X_validation = np.array([x1_u_val, x2_u_val]).T
        self.T = np.array([t_u]).T  # Targets
        self.T_val = np.array([t_u_val]).T
        self.w1 = []
        self.w2 = []
        self.w3 = []
        self.t1 = []
        self.t2 = []
        self.t3 = 0
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0
        self.M1 = 5
        self.M2 = 5
        self.C = 1.00  # Classification error
        self.O = []
        self.O_val = []
        self.e = 0  # Current epoch
        self.V_j = []
        self.V_i = []
        self.Vj_val = []
        self.Vi_val = []
        self.delta_t1 = 0
        self.delta_t2 = 0
        self.delta_t3 = 0
        self.delta_w1 = 0
        self.delta_w2 = 0
        self.delta_w3 = 0

    def initialize_weights(self):
        size1 = (self.M1, 2)
        self.w1 = np.random.uniform(-0.2, 0.2, size1)

        size2 = (self.M2, self.M1)
        self.w2 = np.random.uniform(-0.2, 0.2, size2)

        size3 = (self.M2, 1)
        self.w3 = np.random.uniform(-0.2, 0.2, size3)

    def initialize_thresholds(self):
        size1 = (self.M1, 1)
        self.t1 = np.random.uniform(-1, 1, size1)

        size2 = (self.M2, 1)
        self.t2 = np.random.uniform(-1, 1, size2)

        self.t3 = np.random.uniform(-1, 1)

    def compute_V_j(self):
        self.V_j = np.zeros((self.M1, length_training))

        sum0 = 0
        for j in range(0, self.M1):
            for k in range(0, 2):
                sum0 = sum0 + np.dot(self.w1[j][k], self.X[:, k]) - self.t1[j][0]
            self.V_j[j, :] = np.tanh(sum0)

    def compute_V_i(self):
        self.V_i = np.zeros((self.M2, length_training))

        V_j_tmp = self.V_j.copy()  # Index error

        sum1 = 0

        for i in range(0, self.M2):
            for j in range(0, self.M1):
                sum1 = sum1 + self.w2[i][j] * V_j_tmp[j][:] - self.t2[i][0]
            self.V_i[i, :] = np.tanh(sum1)

    def compute_output(self):
        V_i_tmp = self.V_i.copy()

        sum2 = 0
        for i in range(0, self.M2):
            sum2 = sum2 + np.dot(self.w3[i][0], V_i_tmp[i][:]) - self.t3
        self.O = np.tanh(sum2).T

    def compute_Vj_val(self):
        self.Vj_val = np.zeros((self.M1, length_validation))

        sum0 = 0
        for j in range(0, self.M1):
            for k in range(0, 2):
                sum0 = sum0 + np.dot(self.w1[j][k], self.X_validation[:, k]) - self.t1[j][0]
            self.Vj_val[j, :] = np.tanh(sum0)

    def compute_Vi_val(self):
        self.Vi_val = np.zeros((self.M2, length_validation))

        V_j_tmp = self.Vj_val.copy()  # Index error

        sum1 = 0

        for i in range(0, self.M2):
            for j in range(0, self.M1):
                sum1 = sum1 + self.w2[i][j] * V_j_tmp[j][:] - self.t2[i][0]
            self.Vi_val[i, :] = np.tanh(sum1)

    def compute_output_val(self):
        V_i_tmp = self.Vi_val.copy()

        sum2 = 0
        for i in range(0, self.M2):
            sum2 = sum2 + np.dot(self.w3[i][0], V_i_tmp[i][:]) - self.t3
        self.O_val = np.tanh(sum2).T

    def compute_classification_error(self):
        target = self.T_val.copy()

        output = self.O_val.copy()
        output[output == 0] = 1
        output = np.sign(output)

        error = (np.subtract(output, target))
        error = np.abs(error)
        error_sum = np.sum(error)

        self.C = error_sum / (2 * p_val)

    def train_network(self):

        self.w1 = np.add(self.w1, np.transpose(self.n * self.delta_w1))
        self.w2 = np.add(self.w2, np.transpose(self.n * self.delta_w2))
        self.w3 = np.add(self.w3, np.transpose(self.n * self.delta_w3))

        self.t1 = np.subtract(self.t1, self.n * self.delta_t1)
        self.t2 = np.subtract(self.t2, self.n * self.delta_t2)
        self.t3 = np.subtract(self.t3, self.n * self.delta_t3)

    def propagate_forward(self):
        self.compute_V_j()
        self.compute_V_i()
        self.compute_output()

    def propagate_backward(self, u0):

        tmp = self.X.copy()
        input_tmp = tmp[u0, :]
        input_tmp = input_tmp.reshape(2, 1)

        delta = (self.T[u0] - self.O[u0]) * g_prime(self.b3)
        delta = delta.reshape((1, 1))
        V_i_tmp = self.V_i[:, u0]
        V_i_tmp = V_i_tmp.reshape((self.M2, 1))
        self.delta_t3 = delta
        self.delta_w3 = np.dot(delta, V_i_tmp.T)

        delta = delta * self.w3 * g_prime(self.b2)
        self.delta_t2 = delta
        self.delta_w2 = np.dot(delta.T, self.V_j[:, u0])

        delta = np.dot(delta.T, self.w2) * g_prime(self.b1)
        self.delta_t1 = delta
        self.delta_w1 = np.dot(input_tmp, delta)


def g_prime(b):
    return 1 - np.tanh(b) ** 2


def main():
    perceptron = TwoLayerPerceptron()

    perceptron.initialize_weights()
    perceptron.initialize_thresholds()

    maximum_epochs = 100
    current_epoch = 0

    while perceptron.C >= 0.12 and current_epoch < maximum_epochs:

        for i in range(0, 100):
            u = np.random.randint(0, length_training)

            perceptron.propagate_forward()
            perceptron.propagate_backward(u)

            perceptron.train_network()

        perceptron.compute_Vj_val()
        perceptron.compute_Vi_val()
        perceptron.compute_output_val()

        errors = np.zeros((length_validation, 1))
        perceptron.compute_classification_error()
        errors[current_epoch] = perceptron.C

        print("Currently on epoch number " + str(current_epoch) + "\nValidation error = " + str(errors[current_epoch]))

        current_epoch += 1

    np.savetxt(os.path.join('.', 'w1.csv'), perceptron.w1, delimiter=',')
    np.savetxt(os.path.join('.', 'w2.csv'), perceptron.w2, delimiter=',')
    np.savetxt(os.path.join('.', 'w3.csv'), perceptron.w3, delimiter=',')
    np.savetxt(os.path.join('.', 't1.csv'), perceptron.t1, delimiter=',')
    np.savetxt(os.path.join('.', 't2.csv'), perceptron.t2, delimiter=',')
    np.savetxt(os.path.join('.', 't3.csv'), perceptron.t3, delimiter=',')


main()
