import tensorflow as tf
import copy

EPS = 1e-10

class StandardScaler(tf.keras.layers.experimental.preprocessing.PreprocessingLayer):
    def __init__(self, input_shape):
        super(StandardScaler, self).__init__()
        assert isinstance(input_shape, list) or isinstance(input_shape, tuple)
        self.build(input_shape=list(input_shape))

    def build(self, input_shape):
        shape = copy.deepcopy(input_shape)
        shape[0] = 1 # The batch shape is set to 1.
        self.mean = self.add_weight(
            name='mean',
            shape=shape,
            dtype=tf.float32,
            trainable=False,
            initializer='zeros'
        )
        self.std = self.add_weight(
            name='std',
            shape=shape,
            dtype=tf.float32,
            trainable=False,
            initializer='ones'
        )
        self.built = True

    def adapt(self, data, reset_state=True):
        mean = tf.reduce_mean(data, axis=0, keepdims=True)
        std = tf.math.reduce_std(data, axis=0, keepdims=True) # maybe use reduce_var?
        self.mean.assign(mean)
        self.std.assign(std)

    def call(self, inputs, **kwargs):
        return (inputs - self.mean) / (self.std + EPS)

    def inverse_call(self, inputs):
        return inputs * self.std + self.mean