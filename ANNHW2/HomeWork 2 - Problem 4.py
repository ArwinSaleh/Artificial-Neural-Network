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

length_training = len(training_set)
length_validation = len(validation_set)


class TwoLayerPerceptron:

    def __init__(self):
        self.M1 = 10
        self.M2 = 10
        self.n = 0.04  # Learning rate
        self.X = np.array([x1_u, x2_u])  # Inputs
        self.X_validation = np.array([x1_u_val, x2_u_val]).T
        self.T = np.array([t_u]).T  # Targets
        self.T_val = np.array([t_u_val]).T
        self.w1 = np.zeros((2, self.M1))
        self.w2 = np.zeros((self.M1, self.M2))
        self.w3 = np.zeros((1, self.M2))
        self.t1 = np.zeros((1, self.M1))
        self.t2 = np.zeros((1, self.M2))
        self.t3 = 0
        self.b1 = np.zeros((self.M1, length_training))
        self.b2 = np.zeros((self.M2, length_training))
        self.b3 = np.zeros((1, length_training))
        self.C = 1.00  # Classification error
        self.O = np.zeros((1, length_training))
        self.O_val = np.zeros((1, length_validation))
        self.e = 0  # Current epoch
        self.V_j = np.zeros((self.M1, length_training))
        self.V_i = np.zeros((self.M2, length_training))
        self.Vj_val = np.zeros((self.M1, length_validation))
        self.Vi_val = np.zeros((self.M2, length_validation))
        self.delta_t1 = np.zeros((1, self.M1))
        self.delta_t2 = np.zeros((1, self.M2))
        self.delta_t3 = 0
        self.delta_w1 = np.zeros((2, self.M1))
        self.delta_w2 = np.zeros((self.M1, self.M2))
        self.delta_w3 = np.zeros((1, self.M2))

    def initialize_weights(self):
        size1 = (2, self.M1)
        self.w1 = np.random.uniform(0, 1, size1)

        size2 = (self.M1, self.M2)
        self.w2 = np.random.uniform(0, 1, size2)

        size3 = (1, self.M2)
        self.w3 = np.random.uniform(0, 1, size3)

    def initialize_thresholds(self):
        size1 = (1, self.M1)
        self.t1 = np.zeros(size1)

        size2 = (1, self.M2)
        self.t2 = np.zeros(size2)

        self.t3 = 0

    def compute_V_j(self):
        tmp = np.dot(self.X.T, self.w1) - self.t1
        self.b1 = tmp
        self.V_j = np.tanh(tmp)

    def compute_V_i(self):
        V_j_tmp = self.V_j.copy()  # Index error
        tmp = np.dot(V_j_tmp, self.w2) - self.t2
        self.b2 = tmp
        self.V_i = np.tanh(tmp)

    def compute_output(self):
        V_i_tmp = self.V_i.copy()
        tmp = np.dot(V_i_tmp, self.w3.T) - self.t3
        self.b3 = tmp
        self.O = np.tanh(tmp)

    def compute_Vj_val(self):
        tmp = np.dot(self.w1.T, self.X_validation.T) - np.transpose(self.t1)
        self.Vj_val = np.tanh(tmp)

    def compute_Vi_val(self):
        V_j_tmp = self.Vj_val.copy()  # Index error
        tmp = np.dot(self.w2, V_j_tmp) - np.transpose(self.t2)
        self.Vi_val = np.tanh(tmp)

    def compute_output_val(self):
        V_i_tmp = self.Vi_val.copy()
        tmp = np.dot(self.w3, V_i_tmp) - self.t3
        self.O_val = np.tanh(tmp).T

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
        self.w1 = np.add(self.w1, self.n * self.delta_w1)
        self.w2 = np.add(self.w2, self.n * self.delta_w2)
        self.w3 = np.add(self.w3, self.n * self.delta_w3)

        self.t1 = np.subtract(self.t1, self.n * self.delta_t1)
        self.t2 = np.subtract(self.t2, self.n * self.delta_t2)
        self.t3 = np.subtract(self.t3, self.n * self.delta_t3)

    def propagate_forward(self):
        self.compute_V_j()
        self.compute_V_i()
        self.compute_output()

    def propagate_backward(self, u0):
        Vi_tmp = [self.V_i[u0, :]]
        Vj_tmp = [self.V_j[u0, :]]
        X_tmp = [self.X[:, u0]]

        delta = (self.T[u0] - self.O[u0]) * g_prime(self.b3, u0)
        self.delta_t3 = delta
        self.delta_w3 = np.dot(delta, Vi_tmp)

        delta = np.dot(delta, self.w3, g_prime(self.b2, u0))
        self.delta_t2 = delta
        self.delta_w2 = np.dot(delta, np.transpose(Vj_tmp))

        delta = np.dot(delta, self.w2, g_prime(self.b1, u0))
        self.delta_t1 = delta
        delta = delta.reshape((self.M2, 1))
        self.delta_w1 = np.dot(np.transpose(X_tmp), delta.T)


def g_prime(b, u):
    return 1 - np.tanh(b[u, :]) ** 2


def main():
    perceptron = TwoLayerPerceptron()

    perceptron.initialize_weights()
    perceptron.initialize_thresholds()

    maximum_epochs = 100
    current_epoch = 0
    while perceptron.C >= 0.12 and current_epoch < maximum_epochs:

        for i in range(0, length_training):
            u = np.random.randint(0, length_training)

            perceptron.propagate_forward()
            perceptron.propagate_backward(u)
            perceptron.train_network()

        perceptron.compute_Vj_val()
        perceptron.compuste_Vi_val()
        perceptron.compute_output_val()
        perceptron.compute_classification_error()

        print("Currently on epoch number " + str(current_epoch * 20) + "\nValidation error = " + str(perceptron.C))

        current_epoch += 1

    np.savetxt(os.path.join('.', 'w1.csv'), perceptron.w1, delimiter=',')
    np.savetxt(os.path.join('.', 'w2.csv'), perceptron.w2, delimiter=',')
    np.savetxt(os.path.join('.', 'w3.csv'), perceptron.w3, delimiter=',')
    np.savetxt(os.path.join('.', 't1.csv'), perceptron.t1, delimiter=',')
    np.savetxt(os.path.join('.', 't2.csv'), perceptron.t2, delimiter=',')
    np.savetxt(os.path.join('.', 't3.csv'), perceptron.t3, delimiter=',')


main()
