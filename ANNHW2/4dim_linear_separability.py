import numpy as np
from numpy import genfromtxt
import pandas as pd
import mpmath 
import math as mth

x_temp = genfromtxt('input_data_numeric.csv', delimiter=',')

index_u = x_temp[:, 0]

x_1 = x_temp[:, 1]

x_2 = x_temp[:, 2]

x_3 = x_temp[:, 3]

x_4 = x_temp[:, 4]

x = np.array([x_1, x_2, x_3, x_4]).T



t_A = [1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1]
t_B = [-1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1]
t_C = [1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1]
t_D = [1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1]
t_E = [-1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1]
t_F = [1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1]


def generate_weights():
    s = (4, 1)

    # Uniformly distributed random values between -0.2 and 0.2
    weight_matrix = np.random.uniform(-0.2, 0.2, size=s)

    return weight_matrix

#def backpropagation():



def output_u(b_u):

    return np.tanh(b_u / 2)


def b_u(theta, weights, x_u):
    return (- theta + np.dot(weights.T, x_u.T))


def delta_w(n,output, target, b_u, x_u):
    return (n * (target - output) * ((1 - b_u**2), x_u)) / 2   # transpose

def delta_theta(n, output, target, b_u):
    return - n * (target - output) * (1 - b_u ** 2) / 2


def stochastic_gradient_descent():

    weights = generate_weights()
    n = 0.02    # Learning rate
    theta = np.random.uniform(-1, 1)    # Threshold

    t = t_A

    iterations = 10**5
    i = 0
    output = np.zeros((16, 1)) 

    
    while i < iterations:
        u = np.random.randint(0, 16)    # random component on index u
        x_u = x[u, :]
        b = b_u(theta, weights, x_u) 
        target = t[u]

        b_temp = b[0]

        #output[u] = output_u(b_temp)


        #if np.sign(output[u]) == np.array(t): 
         #       i = iterations
          #      print("Linearly separable")


        d_theta = delta_theta(n, output[u], target, b)
        d_weights = delta_w(n, output[u], target, b, x_u)

        theta = theta + d_theta
        weights = weights + d_weights

        i = i + 1
        
        if i == iterations:
            print("Not linearly separable")


def test():

    weights = generate_weights()
    n = 0.02    # Learning rate
    theta = np.random.uniform(-1, 1)    # Threshold

    t = t_A

    iterations = 10**5
    i = 0
    output = np.zeros((16, 1)) 
    

    u = np.random.randint(0, 16)    # random component on index u
    x_u = x[u, :]
    b = b_u(theta, weights, x_u) 
    target = t[u]

    o_temp = output_u(b)

    output[u] = o_temp

    print(o_temp[0])

stochastic_gradient_descent()