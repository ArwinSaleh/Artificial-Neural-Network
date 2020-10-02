import numpy as np
from numpy import genfromtxt

training_set = genfromtxt('training_set.csv', delimiter=',')
validation_set = genfromtxt('validation_set.csv', delimiter=",")

x1_u = training_set[:, 0]
x2_u = training_set[:, 1]
t_u = training_set[:, 2]

x1_u_val = validation_set[:, 0]
x2_u_val = validation_set[:, 1]
t_u_val = validation_set[:, 2]

u_index = len(t_u)
p_val = len(t_u_val)


class TwoLayerPerceptron:

    def __init__(self):
        self.X = np.array([x1_u, x2_u]).T
        self.T = np.array([t_u]).T
        self.w1 = []
        self.w2 = []
        self.w3 = []
        self.t1 = []
        self.t2 = []
        self.t3 = 0
        self.M1 = 10
        self.M2 = 10
        self.C = 0.00
        self.O = np.zeros((u_index, 1))

        self.V_j = []
        self.V_i = []

    def initialize_weights(self):
        size1 = (self.M1, 2)
        self.w1 = np.random.uniform(-0.2, 0.2, size1)

        size2 = (self.M2, self.M1)
        self.w2 = np.random.uniform(-0.2, 0.2, size2)

        size3 = (self.M2, 1)
        self.w3 = np.random.uniform(-0.2, 0.2, size3)

    def initialize_thresholds(self):
        size1 = (self.M1, 1)
        self.t1 = np.zeros(size1)

        size2 = (self.M2, 1)
        self.t2 = np.zeros(size2)

        self.t3 = 0

    def compute_V_j(self):
        self.V_j = np.zeros((self.M1, u_index))

        sum0 = 0
        for j in range(0, self.M1):
            for k in range(0, 2):
                sum0 = sum0 + self.w1[j][k] * self.X[:, k] - self.t1[j]
            self.V_j[j, :] = np.tanh(sum0)

        return self.V_j

    def compute_V_i(self):
        self.V_i = np.zeros((self.M2, u_index))

        V_j_tmp = self.compute_V_j().copy()  # Index error

        sum1 = 0

        for i in range(0, self.M2):
            for j in range(0, self.M1):
                sum1 = sum1 + self.w2[i][j] * V_j_tmp[j][:] - self.t2[i]
            self.V_i[i, :] = np.tanh(sum1)

        return self.V_i

    def compute_output(self):
        V_i_tmp = self.compute_V_i().copy()

        sum2 = 0
        for i in range(0, self.M2):
            sum2 = sum2 + self.w3[i].T * V_i_tmp[i][:] - self.t3
        self.O = np.tanh(sum2)

        return self.O

    def classification_error(self):
        target = self.T.copy()

        output = self.compute_output().copy()
        output[output == 0] = 1
        output = np.sign(output)

        error = np.abs(output - target.T)

        self.C = sum(error.T) / (2 * p_val)

    def train_network(self):

        self.classification_error()

        self.t1 = self.t1 - self.C * self.t1
        self.t2 = self.t2 - self.C * self.t2
        self.t3 = self.t3 - self.C * self.t3


if __name__ == '__main__':

    perceptron = TwoLayerPerceptron()

    perceptron.initialize_weights()
    perceptron.initialize_thresholds()

    print(perceptron.compute_output())
