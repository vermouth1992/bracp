import tensorflow as tf
import tensorflow_probability as tfp
from rlutils.tf.distributions import make_independent_normal_from_params
from rlutils.tf.nn.functional import build_mlp

from .base import Flow, SequentialFlow, ConditionalFlowModel

tfd = tfp.distributions
tfl = tfp.layers


class AffineCouplingFlow(Flow):
    def __init__(self, input_dim, parity, mlp_hidden=64, num_layers=3):
        super(AffineCouplingFlow, self).__init__()
        assert input_dim > 1, f'input_dim must be greater than 1. Got {input_dim}'
        self.parity = parity
        self.num_layers = num_layers
        self.scale = tf.Variable(initial_value=0., dtype=tf.float32, trainable=True, name='scale')
        self.scale_shift = tf.Variable(initial_value=0., dtype=tf.float32, trainable=True, name='scale_shift')
        self.left_dim = input_dim // 2
        self.right_dim = input_dim - self.left_dim
        if not self.parity:
            self.net = build_mlp(input_dim=self.left_dim, output_dim=self.right_dim * 2, mlp_hidden=mlp_hidden,
                                 batch_norm=False, num_layers=self.num_layers)
        else:
            self.net = build_mlp(input_dim=self.right_dim, output_dim=self.left_dim * 2, mlp_hidden=mlp_hidden,
                                 batch_norm=False, num_layers=self.num_layers)

    def call(self, x, training=None, mask=None):
        x0, x1 = x[:, :self.left_dim], x[:, self.left_dim:]
        if self.parity:
            x0, x1 = x1, x0
        s, t = tf.split(self.net(x0, training=training), 2, axis=-1)
        log_s = tf.tanh(s) * self.scale + self.scale_shift
        z0 = x0
        z1 = tf.exp(log_s) * x1 + t
        if self.parity:
            z0, z1 = z1, z0
        z = tf.concat((z0, z1), axis=-1)
        return z, tf.reduce_sum(log_s, axis=-1)

    def backward(self, z, training=None):
        z0, z1 = z[:, :self.left_dim], z[:, self.left_dim:]
        if self.parity:
            z0, z1 = z1, z0
        s, t = tf.split(self.net(z0, training=training), 2, axis=-1)
        log_s = tf.tanh(s) * self.scale + self.scale_shift
        x0 = z0
        x1 = (z1 - t) * tf.exp(-log_s)
        if self.parity:
            x0, x1 = x1, x0
        x = tf.concat((x0, x1), axis=-1)
        return x


class RealNVP(SequentialFlow):
    def __init__(self, x_dim, num_layers=4, num_layers_coupling=3, mlp_hidden=64, lr=1e-3):
        self.x_dim = x_dim
        self.num_layers = num_layers
        self.mlp_hidden = mlp_hidden
        self.num_layers_coupling = num_layers_coupling
        super(RealNVP, self).__init__(lr=lr)
        self.build(input_shape=tf.TensorShape([None, self.x_dim]))

    def _make_flow(self):
        flows = []
        for _ in range(self.num_layers):
            flows.append(AffineCouplingFlow(input_dim=self.x_dim, parity=True, mlp_hidden=self.mlp_hidden,
                                            num_layers=self.num_layers_coupling))
            flows.append(AffineCouplingFlow(input_dim=self.x_dim, parity=False, mlp_hidden=self.mlp_hidden,
                                            num_layers=self.num_layers_coupling))
        return flows

    def _make_prior(self):
        return tfd.Independent(tfd.Normal(loc=tf.zeros(shape=self.x_dim), scale=tf.ones(shape=self.x_dim)),
                               reinterpreted_batch_ndims=1)


class ConditionalRealNVP(RealNVP, ConditionalFlowModel):
    """
    We consider conditional flow via fixed prior dependent on x from x -> y' -> y.
    Note that we assume the input actions are raw actions before Tanh!
    """

    def __init__(self, x_dim, y_dim, mlp_hidden=128, num_layers=4, lr=1e-3, num_layers_coupling=3):
        self.y_dim = y_dim
        super(ConditionalRealNVP, self).__init__(x_dim=x_dim, mlp_hidden=mlp_hidden,
                                                 num_layers=num_layers, lr=lr,
                                                 num_layers_coupling=num_layers_coupling)

    def _make_prior(self):
        model = build_mlp(input_dim=self.x_dim, output_dim=self.y_dim * 2, mlp_hidden=self.mlp_hidden,
                          num_layers=self.num_layers_coupling, batch_norm=False)
        model.add(tfl.DistributionLambda(make_distribution_fn=make_independent_normal_from_params,
                                         convert_to_tensor_fn=tfd.Distribution.sample))
        return model
