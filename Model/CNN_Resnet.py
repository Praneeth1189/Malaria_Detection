import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import InputLayer
from keras import layers
from Classifier import Classifier
from ResNetBlock import ResNetBlock
from UpDownBlock import UpDownBlock

class model(layers.Layer):
    def __init__(self):
        super().__init__()
        pass
    def __call__(self,x):
        self.model=Sequential([
                    InputLayer(input_shape=(224,224,3)),
                    ResNetBlock(8,3,1,'relu'),
                    UpDownBlock(),
                    ResNetBlock(16,3,1,'relu'),
                    UpDownBlock(),
                    ResNetBlock(32,3,1,'relu'),
                    UpDownBlock(),
                    ResNetBlock(64,3,1,'relu'),
                    UpDownBlock(),
                    Classifier()
                    ])
        return self.model(x)
