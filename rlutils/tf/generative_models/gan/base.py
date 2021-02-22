import tensorflow as tf
import tensorflow_probability as tfp

from rlutils.tf.functional import compute_accuracy

tfd = tfp.distributions


class GAN(tf.keras.Model):
    def __init__(self, n_critics=5, noise_dim=100):
        super(GAN, self).__init__()
        self.n_critics = n_critics
        self.noise_dim = noise_dim
        self.generator = self._make_generator()
        self.discriminator = self._make_discriminator()
        self.prior = self._make_prior()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.logger = None

    def compile(self, generator_optimizer, discriminator_optimizer):
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        super(GAN, self).compile()

    @tf.function
    def generate(self, z):
        return self.generator(z, training=False)

    def _make_prior(self):
        return tfd.Independent(tfd.Normal(loc=tf.zeros(self.noise_dim), scale=tf.ones(self.noise_dim)),
                               reinterpreted_batch_ndims=1)

    def _make_generator(self) -> tf.keras.Model:
        raise NotImplementedError

    def _make_discriminator(self) -> tf.keras.Model:
        raise NotImplementedError

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        self.logger.log_tabular('GenLoss', average_only=True)
        self.logger.log_tabular('DiscLoss', average_only=True)

    @tf.function
    def sample(self, n):
        print(f'Tracing sample with n={n}')
        noise = self.prior.sample(n)
        outputs = self.generator(noise, training=False)
        return outputs

    def predict_real_fake(self, x):
        print(f'Tracing predict_real_fake with x={x}')
        return tf.sigmoid(self.discriminator(x, training=False))

    def _discriminator_loss(self, outputs):
        real_output, fake_output = outputs
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def _generator_loss(self, outputs):
        return self.cross_entropy(tf.ones_like(outputs), outputs)

    @tf.function
    def _train_generator(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = self.prior.sample(batch_size)
        with tf.GradientTape() as tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = self._generator_loss(fake_output)
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        return gen_loss

    @tf.function
    def _train_discriminator(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = self.prior.sample(batch_size)
        generated_images = self.generator(noise, training=True)
        with tf.GradientTape() as tape:
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            disc_loss = self._discriminator_loss(outputs=(real_output, fake_output))
        grads = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return disc_loss

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        gen_loss = self._train_generator(x)
        disc_loss = self._train_discriminator(x)
        return {
            'gen_loss': gen_loss,
            'disc_loss': disc_loss
        }


class ACGAN(GAN):
    def __init__(self, num_classes, class_loss_weight=1., *args, **kwargs):
        self.num_classes = num_classes
        self.class_loss_weight = class_loss_weight
        super(ACGAN, self).__init__(*args, **kwargs)

    @tf.function
    def sample_with_labels(self, labels):
        noise = self.prior.sample(labels.shape[0])
        return self.generate(z=(noise, labels))

    @tf.function
    def sample(self, n):
        labels = tf.random.uniform(shape=(n,), minval=0, maxval=self.num_classes, dtype=tf.int32)
        return self.sample_with_labels(labels=labels)

    def _compute_classification_loss(self, logits, labels):
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        return tf.reduce_mean(loss, axis=0)

    def _generator_loss(self, outputs):
        fake_output, fake_logits, fake_labels = outputs
        validity_loss = super(ACGAN, self)._generator_loss(fake_output)
        classification_loss = self._compute_classification_loss(fake_logits, fake_labels)
        loss = validity_loss + classification_loss * self.class_loss_weight
        return loss

    def _discriminator_loss(self, outputs):
        real_output, fake_output, real_logits, real_labels = outputs
        validity_loss = super(ACGAN, self)._discriminator_loss(outputs=(real_output, fake_output))
        classification_loss = self._compute_classification_loss(real_logits, real_labels)
        loss = validity_loss + classification_loss * self.class_loss_weight
        return loss

    @tf.function
    def _train_generator(self, data):
        real_images, real_labels = data
        batch_size = tf.shape(real_images)[0]
        noise = self.prior.sample(batch_size)
        with tf.GradientTape() as tape:
            generated_images = self.generator(inputs=(noise, real_labels), training=True)
            fake_output, fake_logits = self.discriminator(generated_images, training=True)
            gen_loss = self._generator_loss(outputs=(fake_output, fake_logits, real_labels))
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        accuracy = compute_accuracy(fake_logits, real_labels)
        return gen_loss, accuracy

    @tf.function
    def _train_discriminator(self, data):
        real_images, real_labels = data
        batch_size = tf.shape(real_images)[0]
        noise = self.prior.sample(batch_size)
        generated_images = self.generator(inputs=(noise, real_labels), training=True)
        with tf.GradientTape() as tape:
            real_output, real_logits = self.discriminator(real_images, training=True)
            fake_output, _ = self.discriminator(generated_images, training=True)
            disc_loss = self._discriminator_loss(outputs=(real_output, fake_output,
                                                          real_logits,
                                                          real_labels))
        grads = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        # compute accuracy
        accuracy = compute_accuracy(real_logits, real_labels)
        return disc_loss, accuracy

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        for _ in range(self.n_critics - 1):
            self._train_discriminator(data=(x, y))
        disc_loss, disc_accuracy = self._train_discriminator(data=(x, y))
        gen_loss, gen_accuracy = self._train_generator(data=(x, y))
        return {
            'gen_loss': gen_loss,
            'gen_acc': gen_accuracy,
            'disc_loss': disc_loss,
            'disc_acc': disc_accuracy
        }

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        _, real_logits = self.discriminator(x, training=False)
        disc_accuracy = compute_accuracy(real_logits, y)
        return {
            'disc_acc': disc_accuracy
        }
