"""
Modified from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
"""

import tensorflow as tf
import numpy as np

class OrnsteinUhlenbeckActionNoise(tf.keras.layers.Layer):
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        super(OrnsteinUhlenbeckActionNoise, self).__init__()
        self.theta = theta
        self.mu = tf.convert_to_tensor(mu, dtype=tf.float32)
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.x_prev = tf.Variable(initial_value=tf.zeros_like(self.mu), trainable=False)
        self.reset()

    def __call__(self, inputs=tf.random.normal(shape=()), **kwargs):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * tf.random.normal(shape=self.mu.shape)
        self.x_prev.assign(x)
        return x

    def reset(self):
        self.x_prev.assign(value=self.x0) if self.x0 is not None else tf.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


if __name__ == '__main__':
    ou = OrnsteinUhlenbeckActionNoise(mu=np.random.normal(size=[5]))
    for _ in range(10):
        print(ou())