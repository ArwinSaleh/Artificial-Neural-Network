import numpy as np
from numpy import genfromtxt
import math as mth

x_temp = genfromtxt('input_data_numeric.csv', delimiter=',')

index_u = x_temp[:, 0]

x_1 = x_temp[:, 1]

x_2 = x_temp[:, 2]

x_3 = x_temp[:, 3]

x_4 = x_temp[:, 4]

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

        self.n = 0.02   # Learning rate
        self.W = generate_weights()
        self.theta = generate_threshold()
        self.X = np.zeros((4,1))
        self.O = np.zeros((16, 1))
        self.T = np.zeros((16, 1))
        self.err = 0
        

    def error(self, u):
        error = (self.T[u] - 0.5 * self.O[u] * self.g_prime())
        self.err = error.item()


    def train(self):
        self.theta = self.theta - 0.02 * self.err
        self.W = self.W + (0.02 * np.array(self.err * self.X)).T


    def generate_b(self):
        x_u = np.array(self.X)
        return -self.theta + np.dot(self.W.T, x_u.T)

    def generate_g(self):
        b_u = self.generate_b()
        return np.tanh(0.5 * b_u)


    def g_prime(self):
        b_u = self.generate_b()
        return (1 - np.tanh(0.5 * b_u) ** 2)


    def output(self, u):
        self.O[u] = self.generate_g()


    def linearly_separable(self):
        output = self.O.copy()
        output[output == 0] = 1
        output = np.sign(output)

        if output.all() == self.T.all():
            return True
    
        else:
            return False

def generate_weights():
    weights = np.random.uniform(-0.2, 0.2, (4, 1))
    return np.asmatrix(weights)


def generate_threshold():
    return np.random.uniform(-1, 1)




def main():

    structure = StochasticGradientDescent()
    maximum_iterations = 10**5
    u_range = 16

    function_names = ['A', 'B', 'C', 'D', 'E', 'F']
    linearly_separable_list = []
    

    name_index = 0

    for t in T_data:  # We go through each input-pattern
        structure.T = t.copy()
        linearly_separable = False
        current_boolean_function = function_names[name_index]
        structure.T = t.copy()


        for i in range(0, 10):   # number of retries
            for j in range(0, maximum_iterations):

                for u in range(0, u_range):
                    structure.X = X_data[u, :]
                    structure.output(u)
                    structure.error(u)
                    structure.train()

                linearly_separable = structure.linearly_separable()

                if linearly_separable:
                    
                    print("The following boolean function is linearly separable: " + current_boolean_function)
                    linearly_separable_list.append(current_boolean_function)
                    break
                
            
            if linearly_separable:
                break
            else:
                print("The following boolean function is NOT linearly separable: " + current_boolean_function)
            
            name_index = name_index + 1

            print("Try number: " + str(j + 1))
            structure.__init__()
            structure.T = t.copy()

        structure.__init__()

    print("The following functions are linearly separable: " + str(linearly_separable_list))


main()


        



