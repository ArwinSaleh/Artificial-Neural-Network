import numpy as np
from numpy import genfromtxt
import pandas as pd
import mpmath 

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

def sgn(i):  # This method is used to define the signum function at signum(0).

    if i == 0:
        return 1
    else:
        return np.sign(i)


def generate_weights():
    s = (16, 1)

    # Uniformly distributed random values between -0.2 and 0.2
    weight_matrix = np.random.uniform(-0.2, 0.2, size=s)

    return weight_matrix

#def backpropagation():




def g_prime(b):
    return (mpmath.sech(b))**2


def output_u(b_u):

    return np.tanh(b_u / 2)


def b_u(theta, weights, x_u):
    return (- theta + weights*x_u)



def delta_w(n,output, target, b_u, x_u):
    return (n * (target - output) * (1 - b_u**2) * x_u) / 2   # transpose

def delta_theta(n, output, target, b_u):
    return - n * (target - output) * (1 - b_u ** 2) / 2


if __name__ == "__main__":

    weights = generate_weights()
    n = 0.02    # Learning rate
    theta = np.random.uniform(-1, 1)    # Threshold

    t = t_A

    iterations = 10**5
    i = 0
    output = np.zeros((16, 5))

    
    while i < iterations:

        u = np.random.randint(0, 16)    # random component on index u
        x_u = x[u, :]
        b = b_u(theta, weights, x_u) 
        target = t[u]
        output = output_u(b)
        
        if (output[u] == t).all():
            i = iterations
            print("Linearly separable")

        d_theta = delta_theta(n, output, target, b)
        d_weights = delta_w(n, output, target, b, x_u)

        theta = theta + d_theta
        weights = weights + d_weights

        i = i + 1
        
        if i == iterations:
            print("Not linearly separable")




