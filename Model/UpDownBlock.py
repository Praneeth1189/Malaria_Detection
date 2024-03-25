import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import MaxPool2D, BatchNormalization
from keras.models import Sequential

class UpDownBlock(layers.Layer):
    def __init__(self):
        super().__init__()
        self.Layer1 = MaxPool2D(pool_size=2, strides=2)
        self.Layer2 = BatchNormalization()
    def __call__(self, x):
        M = Sequential([self.Layer1, self.Layer2])
        return M(x)