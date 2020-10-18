import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras import activations
from tensorflow.python.keras.backend import relu, softmax
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.optimizers import SGD



#image_input_layer = network1.fit(validation_data=(val_X, val_y))

class Network1:

    def __init__(self):
        (self.train_X, self.train_y), (self.test_X, self.test_y) = mnist.load_data() 
        self.convolution2d_layer = layers.Conv2D(20, strides=(1,1), activation='relu', padding=(1,1), kernel_size=(5, 5))
        self.max_pooling2d_layer = layers.MaxPooling2D(padding=(0,0), strides=(2,2), pool_size=(2, 2))
        self.fully_connected_layer1 = layers.Dense(100, activation='relu')
        self.fully_connected_layer2 = layers.Dense(10, activation='softmax')
        self.classification_layer = layers.Dense(10)

    def define_network(self):
        self.network1 = models.Sequential()
        self.network1.add(self.convolution2d_layer)
        self.network1.add(BatchNormalization())
        self.network1.add(self.max_pooling2d_layer)
        self.network1.add(Flatten())
        self.network1.add(self.fully_connected_layer1)
        self.network1.add(BatchNormalization())
        self.network1.add(self.fully_connected_layer2)
        self.network1.add(self.classification_layer)
        self.network1.compile(metrics=['accuracy'], loss='categorical_crossentropy')
        return self.network1
    
    def summary(self):
        print(self.network1.summary())

    def restructure_data(self):
        training_shape = (self.train_X.shape[0], 28, 28, 1 )
        testing_shape = (self.test_X.shape[0], 28, 28, 1 )
        self.train_X = self.train_X.reshape(training_shape)
        self.test_X = self.test_X.reshape(testing_shape)
        self.train_y = to_categorical(self.train_y)
        self.test_y = to_categorical(self.test_y)
        return self.train_X, self.train_y, self.test_X, self.test_y