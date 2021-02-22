import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class BetaVAE(tf.keras.Model):
    def __init__(self, latent_dim, beta=1.):
        super(BetaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = self._make_encoder()
        self.decoder = self._make_decoder()
        self.prior = self._make_prior()
        self.logger = None

    def _make_encoder(self) -> tf.keras.Model:
        raise NotImplementedError

    def _make_decoder(self) -> tf.keras.Model:
        raise NotImplementedError

    def _make_prior(self):
        return tfd.Independent(tfd.Normal(loc=tf.zeros(shape=[self.latent_dim], dtype=tf.float32),
                                          scale=tf.ones(shape=[self.latent_dim], dtype=tf.float32)),
                               reinterpreted_batch_ndims=1)

    def encode_distribution(self, inputs):
        return self.encoder(inputs, training=False)

    def encode_sample(self, inputs):
        encode_distribution = self.encode_distribution(inputs)
        encode_sample = encode_distribution.sample()
        return encode_sample

    def encode_mean(self, inputs):
        encode_distribution = self.encode_distribution(inputs)
        encode_sample = encode_distribution.mean()
        return encode_sample

    def decode_distribution(self, z):
        return self.decoder(z, training=False)

    def decode_sample(self, z):
        decode_distribution = self.decode_distribution(z)
        decode_sample = decode_distribution.sample()
        return decode_sample

    def decode_mean(self, z):
        decode_distribution = self.decoder(z, training=False)
        decode_sample = decode_distribution.mean()
        return decode_sample

    @tf.function
    def elbo(self, inputs):
        assert self.beta == 1., 'Only Beta=1.0 has ELBO'
        nll, kld = self(inputs, training=False)
        elbo = -nll - kld
        return elbo

    def sample(self, full_path=True):
        z = self.prior.sample()
        mean = self.decode_mean(z)
        sample = self.decode_sample(z)
        return tf.cond(full_path, true_fn=lambda: sample, false_fn=lambda: mean)

    def call(self, inputs, training=None, mask=None):
        print(f'Tracing _forward with input {inputs}')
        posterior = self.encoder(inputs, training=training)
        encode_sample = posterior.sample()
        out = self.decoder(encode_sample, training=training)
        log_likelihood = out.log_prob(inputs)  # (None,)
        kl_divergence = tfd.kl_divergence(posterior, self.prior)
        # print(f'Shape of nll: {log_likelihood.shape}, kld: {kl_divergence.shape}')
        return -log_likelihood, kl_divergence

    def train_step(self, data):
        data = data[0]
        with tf.GradientTape() as tape:
            nll, kld = self(data, training=True)
            loss = nll + kld * self.beta
            loss = tf.reduce_mean(loss, axis=0)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {
            'loss': loss,
            'nll': nll,
            'kld': kld
        }

    def test_step(self, data):
        data = data[0]
        nll, kld = self(data, training=False)
        loss = nll + kld * self.beta
        loss = tf.reduce_mean(loss, axis=0)
        return {
            'loss': loss,
            'nll': nll,
            'kld': kld
        }

    @tf.function
    def train_on_batch(self,
                       x,
                       y=None,
                       sample_weight=None,
                       class_weight=None,
                       reset_metrics=True,
                       return_dict=False):
        return self.train_step(data=(x,))

    @tf.function
    def test_on_batch(self,
                      x,
                      y=None,
                      sample_weight=None,
                      reset_metrics=True,
                      return_dict=False):
        return self.test_step(data=(x,))


class ConditionalBetaVAE(BetaVAE):
    """
    x + cond -> z + cond -> x
    Encoder should take in (x, cond).
    """

    def call(self, inputs, training=None, mask=None):
        x, cond = inputs
        print(f'Tracing _forward with x={x}, cond={cond}')
        posterior = self.encoder(inputs=(x, cond), training=training)
        encode_sample = posterior.sample()
        out = self.decoder((encode_sample, cond), training=training)
        log_likelihood = out.log_prob(x)  # (None,)
        kl_divergence = tfd.kl_divergence(posterior, self.prior)
        return -log_likelihood, kl_divergence

    def sample(self, cond, full_path=True):
        print(f'Tracing sample with cond={cond}')
        z = self.prior.sample(sample_shape=tf.shape(cond)[0])  # (None, z_dim)
        out_dist = self.decode_distribution(z=(z, cond))
        return tf.cond(full_path, true_fn=lambda: out_dist.sample(), false_fn=lambda: out_dist.mean())

    def sample_n(self, cond, n, full_path=True):
        print(f'Tracing sample with cond={cond}, n={n}')
        batch_size = tf.shape(cond)[0]
        cond = tf.tile(cond, (n, 1))
        samples = self.sample(cond, full_path=full_path)  # (None * n, y_dim)
        shape = [tf.shape(samples)[k] for k in range(tf.rank(samples))]
        return tf.reshape(samples, shape=[n, batch_size] + shape[1:])
