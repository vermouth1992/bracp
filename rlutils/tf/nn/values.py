import tensorflow as tf

from rlutils.tf.nn.functional import build_mlp


class EnsembleMinQNet(tf.keras.Model):
    def __init__(self, ob_dim, ac_dim, mlp_hidden, num_ensembles=2, num_layers=3):
        super(EnsembleMinQNet, self).__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.mlp_hidden = mlp_hidden
        self.num_ensembles = num_ensembles
        self.num_layers = num_layers
        self.q_net = build_mlp(input_dim=self.ob_dim + self.ac_dim,
                               output_dim=1,
                               mlp_hidden=self.mlp_hidden,
                               num_ensembles=self.num_ensembles,
                               num_layers=num_layers,
                               squeeze=True)
        self.build(input_shape=[(None, ob_dim), (None, ac_dim)])

    def get_config(self):
        config = super(EnsembleMinQNet, self).get_config()
        config.update({
            'ob_dim': self.ob_dim,
            'ac_dim': self.ac_dim,
            'mlp_hidden': self.mlp_hidden,
            'num_ensembles': self.num_ensembles,
            'num_layers': self.num_layers
        })
        return config

    def call(self, inputs, training=None, mask=None):
        obs, act = inputs
        inputs = tf.concat((obs, act), axis=-1)
        inputs = tf.tile(tf.expand_dims(inputs, axis=0), (self.num_ensembles, 1, 1))
        q = self.q_net(inputs)  # (num_ensembles, None)
        if training:
            return q
        else:
            return tf.reduce_min(q, axis=0)


class AtariQNetworkDeepMind(tf.keras.Model):
    def __init__(self, act_dim, frame_stack=4, dueling=False, data_format='channels_first', scale_input=True):
        super(AtariQNetworkDeepMind, self).__init__()
        if data_format == 'channels_first':
            self.batch_input_shape = (None, frame_stack, 84, 84)
        else:
            self.batch_input_shape = (None, 84, 84, frame_stack)
        self.features = tf.keras.Sequential([
            tf.keras.layers.InputLayer(batch_input_shape=self.batch_input_shape),
            tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, padding='same', activation='relu',
                                   data_format=data_format),
            tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation='relu',
                                   data_format=data_format),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                                   data_format=data_format),
            tf.keras.layers.Flatten()
        ])
        self.dueling = dueling
        self.scale_input = scale_input
        self.q_feature = tf.keras.layers.Dense(units=512, activation='relu')
        self.adv_fc = tf.keras.layers.Dense(units=act_dim)
        if self.dueling:
            self.value_fc = tf.keras.layers.Dense(units=1)
        else:
            self.value_fc = None
        self.build(input_shape=self.batch_input_shape)

    def call(self, inputs, training=None):
        if self.scale_input:
            # this assumes the inputs is in image format (None, frame_stack, 84, 84)
            inputs = tf.cast(inputs, dtype=tf.float32) / 255.
        features = self.features(inputs, training=training)
        q_value = self.q_feature(features, training=training)
        adv = self.adv_fc(q_value)  # (None, act_dim)
        if self.dueling:
            adv = adv - tf.reduce_mean(adv, axis=-1, keepdims=True)
            value = self.value_fc(q_value)
            q_value = value + adv
        else:
            q_value = adv
        return q_value
