import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Flatten
class Classifier(layers.Layer):
    def __init__(self):
        super().__init__()
        self.Layer1 = Dense(units = 64, activation = "relu")
        self.Layer2 = Dense(units = 16, activation = "relu")
        self.Layer3 = Dense(units = 8, activation = "relu")
        self.Layer4 = Dense(units = 1, activation = "sigmoid")

    def __call__(self,x):
        L = Sequential([Flatten(),
                        self.Layer1,
                        self.Layer2,
                        self.Layer3,
                        self.Layer4])
        return L(x)