import tensorflow as tf
import tensorflow_probability as tfp

from rlutils.future.optimizer import get_adam_optimizer

tfd = tfp.distributions
tfl = tfp.layers

eps = 1e-6


class Flow(tf.keras.Model):
    """
    A flow is a function f that defines a forward (call) and backward path
    """

    def call(self, x, training=None, mask=None):
        raise NotImplementedError

    def backward(self, z, training):
        raise NotImplementedError


class SequentialFlow(Flow):
    """
    A sequence of flows. It is a flow by itself.
    """

    def __init__(self, lr=1e-3):
        super(SequentialFlow, self).__init__()
        self.flows = self._make_flow()
        self.prior = self._make_prior()
        self.logger = None
        self.compile(optimizer=get_adam_optimizer(lr=lr))

    def _make_flow(self):
        raise NotImplementedError

    def _make_prior(self):
        raise NotImplementedError

    def call(self, x, training=None, mask=None):
        z, log_det = x, tf.zeros(shape=tf.shape(x)[0], dtype=tf.float32)
        for flow in self.flows:
            z, delta_log_det = flow(z, training=training)
            log_det += delta_log_det
        return z, log_det

    def backward(self, z, training):
        x = z
        for flow in reversed(self.flows):
            x = flow.backward(x, training=training)
        return x

    def infer(self, x):
        return self(x, training=False)[0]

    def log_prob(self, x, training=False):
        print(f'Tracing log_prob with x:{x}, training:{training}')
        z, log_det = self(x, training=training)
        return log_det + self.prior.log_prob(z)

    def sample(self, n):
        print(f'Tracing sample with n:{n}')
        z = self.prior.sample(sample_shape=n)
        x = self.backward(z, training=False)
        return x

    def _forward(self, x, training):
        print(f'Tracing _forward with x:{x}, training:{training}')
        loss = tf.reduce_mean(-self.log_prob(x, training=training), axis=0)
        return loss

    def train_step(self, data):
        print(f'Tracing train_step with data:{data}')
        with tf.GradientTape() as tape:
            loss = self._forward(data, training=True)
            final_loss = loss + sum(self.losses)
        gradients = tape.gradient(final_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {'loss': loss}

    def test_step(self, data):
        loss = self._forward(data, training=False)
        return {'loss': loss}


class ConditionalFlowModel(SequentialFlow):
    def __init__(self, lr=1e-3):
        super(ConditionalFlowModel, self).__init__(lr=lr)

    def log_prob(self, data, training=False):
        x, y = data
        print(f'Tracing log_prob with x:{x}, y:{y}, training:{training}')
        prior = self.prior(x, training=training)
        z, log_det = self(y, training=training)
        return log_det + prior.log_prob(z)

    def sample(self, x):
        print(f'Tracing sample with x:{x}')
        z = self.prior(x, training=False).sample()
        y = self.backward(z, training=False)
        return y

    def sample_n(self, x, n):
        print(f'Tracing sample with x:{x}, n:{n}')
        x = tf.tile(x, (n, 1))
        output = self.sample(x)
        return tf.reshape(output, shape=(n, -1, self.y_dim))
