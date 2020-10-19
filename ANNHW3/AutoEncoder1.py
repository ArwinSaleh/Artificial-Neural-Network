from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.datasets import mnist
from tensorflow.python.eager.context import PhysicalDevice, device
from tensorflow.python.keras import activations
from tensorflow.python.keras.backend import relu, softmax
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from numpy import mean, std

class Encoder:
    def __init__(self):
        (self.train_X, self.train_y), (self.test_X, self.test_y) = mnist.load_data() 
        self.image_input_layer = layers.InputLayer(input_shape=(28, 28, 1))
        self.fully_connected_layer1 = layers.Dense(50, activation='relu')
        self.fully_connected_layer2 = layers.Dense(2, 'relu')   # Bottleneck layer
        self.fully_connected_layer3 = layers.Dense(784, activation='relu')
        
        
    
    def load_mnist(self):
        training_shape = (self.train_X.shape[0], 28, 28, 1) # Single channel shape
        testing_shape = (self.test_X.shape[0], 28, 28, 1)
        self.train_X = self.train_X.reshape(training_shape)
        self.test_X = self.test_X.reshape(testing_shape)
        self.train_y = to_categorical(self.train_y)
        self.test_y = to_categorical(self.test_y)