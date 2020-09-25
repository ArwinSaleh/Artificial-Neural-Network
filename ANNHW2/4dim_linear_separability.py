import numpy as np
from numpy import genfromtxt
import pandas as pd

x = genfromtxt('input_data_numeric.csv', delimiter=',')

index_u = x[:, 0]

x_1 = x[:, 1]

x_2 = x[:, 2]

x_3 = x[:, 3]

x_4 = x[:, 4]

t_A = [1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1]
t_B = [-1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1]
t_C = [1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1]
t_D = [1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1]
t_E = [-1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1]
t_F = [1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1]


def generate_weights():
    s = (16, 1)

    # Uniformly distributed random values between -0.2 and 0.2
    weight_matrix = np.random.uniform(-0.2, 0.2, size=s)

    return weight_matrix


def output_function(u, theta, weights_i, x_i_u):

    sum = 0
    for i in range(1, 4):
        sum = sum + np.dot(weights_i[i-1], x_i_u[u, i])

    return np.tanh(0.5 * (- theta + sum))


def energy_function(theta, weights_i, x_i_u):
    sum = 0
    for i in range(0, len(index_u)):
        sum = sum + (t_A[i] - output_function(i, theta, weights_i, x_i_u)) ** 2

    return 0.5 * sum

#def stochastic_gradient_descent():




def main():

    weights = generate_weights()

    n = 0.2    # Learning rate
    theta = np.random.uniform(-1, 1)    # Threshold

    H = energy_function(theta, weights, x)



main()
