# Custom L1 distance layer module

# Import dependendencies
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Custom L1 distance layer from jupyter
class L1Dist(Layer):

    #Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # Magic happens here - similarity calculation
    def call(self, inputs_embedding, validation_embedding):
        return tf.math.abs(inputs_embedding - validation_embedding)