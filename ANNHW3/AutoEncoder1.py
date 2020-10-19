import keras
from keras import layers
from keras.datasets import mnist
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt


class AutoEncoder:
    def __init__(self):

        (self.train_X, _), (self.test_X, _) = mnist.load_data() 

        self.image_input_layer = keras.Input(shape=(784,))
        self.fully_connected_layer1 = layers.Dense(50, activation='relu')(self.image_input_layer)
        self.fully_connected_layer2 = layers.Dense(2, 'relu')(self.fully_connected_layer1)   # Bottleneck layer
        self.fully_connected_layer3 = layers.Dense(784, activation='relu')(self.fully_connected_layer2)
        self.auto_encoder = keras.Model(self.image_input_layer, self.fully_connected_layer3)
        self.encoder = keras.Model(self.image_input_layer, self.fully_connected_layer1)

        input_encode = keras.Input(shape=(2,))
        layer_decode = self.auto_encoder.layers[-1]
        self.decoder = keras.Model(input_encode, layer_decode(input_encode))

    def load_mnist(self):
        self.train_X = self.train_X.astype('float32') / 255.0
        self.test_X = self.test_X.astype('float32') / 255.0
        self.train_X = self.train_X.reshape(len(self.train_X), np.prod(self.train_X.shape[1:]))
        self.test_X = self.test_X.reshape(len(self.test_X), np.prod(self.test_X.shape[1:]))

    def train_encoder(self):
        self.auto_encoder.compile(optimizer='adam', loss='binary_crossentropy')
        self.auto_encoder.fit(  self.train_X, self.train_X, 
                                epochs=800, batch_size=8192,
                                shuffle=True, validation_data=(self.test_X, self.test_X), 
                                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')] )

    def plot_information(self):

        encoded = self.encoder.predict(self.test_X)
        decoded = self.decoder.predict(encoded)

        plt.figure(figsize=(10, 2))
        for i in range (5):
            axis = plt.subplot(2, 5, i + 1)
            plt.imshow(self.test_X[i].reshape(28, 28))
            plt.gray()
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)

            axis = plt.subplot(2, 5, i + 6)
            plt.imshow(decoded[i].reshape(28, 28))
            plt.gray()
            axis.get_xaxis.set_visible(False)
            axis.get_yaxis.set_visible(False)
        
        plt.show()

def main():
    encoder = AutoEncoder()
    encoder.load_mnist()
    encoder.train_encoder()
    encoder.plot_information()

main()

        
