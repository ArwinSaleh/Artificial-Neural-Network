import imp
import re
from numpy.lib import tests
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras import activations
from tensorflow.python.keras.backend import relu, softmax
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt


class Network1:

    def __init__(self):
        (self.train_X, self.train_y), (self.test_X, self.test_y) = mnist.load_data() 
        self.image_input_layer = layers.InputLayer(input_shape=(28, 28, 1))
        self.convolution2d_layer = layers.Conv2D(20, strides=(1,1), activation='relu', padding='valid', kernel_size=(5, 5))
        self.max_pooling2d_layer = layers.MaxPooling2D(strides=(2,2), pool_size=(2, 2))
        self.fully_connected_layer1 = layers.Dense(100, activation='relu')
        self.fully_connected_layer2 = layers.Dense(10, activation='softmax')
        self.classification_layer = layers.Dense(10)
        self.optimizer = SGD(momentum = 0.9, lr=0.001)
        self.history_list = list()
        self.score_list = list()

    def define_network(self):
        self.network1 = models.Sequential()
        self.network1.add(self.image_input_layer)
        self.network1.add(self.convolution2d_layer)
        self.network1.add(BatchNormalization())
        self.network1.add(self.max_pooling2d_layer)
        self.network1.add(Flatten())
        self.network1.add(self.fully_connected_layer1)
        self.network1.add(BatchNormalization())
        self.network1.add(self.fully_connected_layer2)
        self.network1.add(self.classification_layer)
        self.network1.compile(metrics=['accuracy'], loss='categorical_crossentropy', optimizer=self.optimizer)
        return self.network1
    
    def summary(self):
        print(self.network1.summary())

    def restructure_data(self):
        training_shape = (self.train_X.shape[0], 28, 28, 1) # Single channel shape
        testing_shape = (self.test_X.shape[0], 28, 28, 1)
        self.train_X = self.train_X.reshape(training_shape)
        self.test_X = self.test_X.reshape(testing_shape)
        self.train_y = to_categorical(self.train_y)
        self.test_y = to_categorical(self.test_y)

    def load_mnist(self):
        self.restructure_data()
        return self.train_X, self.train_y, self.test_X, self.test_y

    def evaluate(self):

        cross_validator = KFold(5, random_state=1, shuffle=True)
        
        for i, j in cross_validator.split(self.train_X):
            net = self.define_network()
            train_X = self.train_X[i]
            train_y = self.train_y[i]
            test_X = self.train_X[j]
            test_y = self.train_y[j]
            history = net.fit(train_X, train_y, validation_data = (test_X, test_y), epochs = 60, batch_size = 8192, verbose = 0)
            _, accuracy = net.evaluate(test_X, test_y, verbose=0)
            self.score_list.append(accuracy)
            self.history_list.append(history)

        return self.history_list, self.score_list

    def plot_information(self):
        for i in range(len(self.history_list)):

            plt.subplot(2, 1, 1)
            plt.title('Loss')
            plt.plot(self.history_list[i].history['loss'], color='orange', label='Training data')
            plt.plot(self.history_list[i].history['val_loss'], color='blue', label ='Testing data')
            
            plt.subplot(2, 1, 2)
            plt.title('Accuracy')
            plt.plot(self.history_list[i].history['accuracy'], color='orange', label='Training data')
            plt.plot(self.history_list[i].history['val_accuracy'], color='blue', label ='Testing data')

        plt.show()
    
    def pixel_scaling(self):

        trainX = self.train_X.astype('float32')
        testX = self.test_X.astype('float32')

        return trainX/255.0, testX/255.0


if __name__ == "__main__":
    net = Network1()
    train_X, train_y, test_X, test_y = net.load_mnist()
    train_X, train_y = net.pixel_scaling()
    history_list, score_list = net.evaluate()
    net.plot_information()