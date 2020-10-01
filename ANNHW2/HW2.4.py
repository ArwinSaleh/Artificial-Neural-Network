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


class TwoLayerPerceptron:

    def __init__(self):
        self.X = np.array([x1_u, x2_u]).T
        self.T = np.array([t_u]).T
        self.w1 = []
        self.w2 = []
        self.w3 = []
        self.t1 = []
        self.t2 = []
        self.t3 = []
        self.M1 = 1
        self.M2 = 1
        self.C = 0
        # self.p_val
        # self.O

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

    def experiment(self):
        input1 = int(input("Choose a value for M1: "))
        self.M1 = input1
        input2 = int(input("Choose a value for M2: "))
        self.M2 = input2
        

    def compute_classification_error(self):


if __name__ == '__main__':
    print("test")
