import numpy as np

class TwoLayerPerceptron:

    def __init__(self):
        #self.X
        self.w1 = []
        self.w2 = []
        #self.w3
        #self.t1
        #self.t2
        #self.t3
        self.M1 =
        self.M2 =
        #self.C
        #self.p_val
        #self.O

    def initialize_w1(self):
        size = (self.M1, 2)
        self.w1 = np.random.uniform(-0.2, 0.2, size)

    def initialize_w2(self):
        size = (self.M2, self.M1)


    def test(self):
        return self.w1

def experiment_M1():
    print("Choose a value for M1")
    value =
    return

def debug():

    struct = TwoLayerPerceptron()
    struct.initialize_w1()
    print(struct.test())

