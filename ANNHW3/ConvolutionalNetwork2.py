import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from numpy import mean, std

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

class Network2:

    def __init__(self):
        (self.train_X, self.train_y), (self.test_X, self.test_y) = mnist.load_data() 
        self.image_input_layer = layers.InputLayer(input_shape=(28, 28, 1))
        self.convolution2d_layer1 = layers.Conv2D(20, strides=(1,1), activation='relu', padding='same', kernel_size=(3, 3), kernel_initializer='he_uniform')
        self.max_pooling2d_layer1 = layers.MaxPooling2D(strides=(2,2), pool_size=(2, 2), padding='valid')
        self.convolution2d_layer2 = layers.Conv2D(30, strides=(1,1), kernel_size=(3,3), padding='same', activation='relu')
        self.max_pooling2d_layer2 = layers.MaxPooling2D(strides=(2,2), pool_size=(2,2), padding='valid')
        self.convolution2d_layer3 = layers.Conv2D(0, strides=(1,1), padding='same', kernel_size=(3,3), activation='relu')
        self.fully_connected_layer = layers.Dense(10, activation='softmax', kernel_initializer='he_uniform')
        self.optimizer = SGD(momentum = 0.9, lr=0.01)
        self.history_list = list()
        self.score_list = list()

    def define_network(self):
        network2 = models.Sequential()
        network2.add(self.image_input_layer)
        network2.add(self.convolution2d_layer1)
        network2.add(BatchNormalization())
        network2.add(self.max_pooling2d_layer1)
        network2.add(Flatten())
        network2.add(self.fully_connected_layer)
        network2.compile(metrics=['accuracy'], loss='categorical_crossentropy', optimizer=self.optimizer)
        return network2
    
    def summary(self):
        print(self.network2.summary())

    def load_mnist(self):
        training_shape = (self.train_X.shape[0], 28, 28, 1) # Single channel shape
        testing_shape = (self.test_X.shape[0], 28, 28, 1)
        self.train_X = self.train_X.reshape(training_shape)
        self.test_X = self.test_X.reshape(testing_shape)
        self.train_y = to_categorical(self.train_y)
        self.test_y = to_categorical(self.test_y)

    def evaluate(self):

        cross_validator = KFold(5, random_state=1, shuffle=True)
        
        for i, j in cross_validator.split(self.train_X):
            net = self.define_network()
            train_X = self.train_X[i]
            train_y = self.train_y[i]
            test_X = self.train_X[j]
            test_y = self.train_y[j]
            history = net.fit(train_X, train_y, validation_data = (test_X, test_y), epochs = 30, batch_size = 8192, verbose = 0)
            _, accuracy = net.evaluate(test_X, test_y, verbose=0)
            print("\nAccuracy = " + str(100 * accuracy))
            self.score_list.append(accuracy)
            self.history_list.append(history)

    def information(self):
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

        print('Mean accuracy: ' + str(100 * mean(self.score_list)))
        print("Std accuracy: " + str(std(self.score_list)))
        print("n = " + str(len(self.score_list)))

    def plot_mean(self):
        plt.boxplot(self.score_list)
        plt.show()

    def pixel_scaling(self):

        train = self.train_X.astype('float32')
        test = self.test_X.astype('float32')

        self.train_X = train / 255.0
        self.test_X = test / 255.0


def main():
    net = Network2()
    net.load_mnist()
    net.pixel_scaling()
    net.evaluate()
    net.information()
    net.plot_mean()

main()