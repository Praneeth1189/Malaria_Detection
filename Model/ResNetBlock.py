import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Conv2D, BatchNormalization

class ResNetBlock(layers.Layer):
    def __init__(self, filters, kernel_size, strides, activation : str):
        super().__init__()
        self.layer1 = Conv2D(kernel_size=kernel_size, filters = filters,
                            padding='same',activation=activation)
        self.layer2 = Conv2D(kernel_size=kernel_size, filters = filters,
                            padding='same',activation=activation)
        self.BatchNorm = BatchNormalization(),

    def __call__(self,x):
        a = self.layer1(x)
        b = self.layer2(a)
        return b+a