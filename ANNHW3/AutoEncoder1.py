import keras
from keras import layers
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.callbacks import TensorBoard


class AutoEncoder:
    def __init__(self):
        (self.train_X, self.train_y), (self.test_X, self.test_y) = mnist.load_data() 
        self.image_input_layer = keras.Input(shape=(784,))
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

    def build_encoder(self):
        self.fully_connected_layer1 = self.fully_connected_layer1(self.image_input_layer)
        self.fully_connected_layer2 = self.fully_connected_layer2(self.fully_connected_layer1)
        self.fully_connected_layer3 = self.fully_connected_layer3(self.fully_connected_layer2)

        auto_encoder = keras.Model(self.image_input_layer, self.fully_connected_layer3)
        auto_encoder.compile(optimizer='adam', loss='binary_crossentropy')

        auto_encoder.fit(self.train_X, self.train_X, epochs=800, batch_size=8192, shuffle=True, validation_data=(self.test_X, self.test_X), callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

def main():
    encoder = AutoEncoder()
    encoder.load_mnist()
    encoder.build_encoder()

        
