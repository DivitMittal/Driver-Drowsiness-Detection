import keras
from keras.layers import Layer


class L1Dist(Layer):
    def call(self, input_embedding, validation_embedding):
        return keras.ops.abs(keras.ops.subtract(input_embedding, validation_embedding))
