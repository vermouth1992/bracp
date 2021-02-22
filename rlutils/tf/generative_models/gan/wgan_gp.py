"""
Improved Training of Wasserstein GANs
"""

import tensorflow as tf
from rlutils.tf.functional import compute_accuracy
from tqdm.auto import tqdm

from .base import GAN, ACGAN


class WassersteinGANGradientPenalty(GAN):
    def __init__(self, gp_weight=10, *args, **kwargs):
        self.gp_weight = gp_weight
        super(WassersteinGANGradientPenalty, self).__init__(*args, **kwargs)

    def _discriminator_loss(self, outputs):
        real_output, fake_output = outputs
        loss = tf.reduce_mean(fake_output, axis=0) - tf.reduce_mean(real_output, axis=0)
        return loss

    def _generator_loss(self, outputs):
        return -tf.reduce_mean(outputs, axis=0)

    def _compute_gp(self, real_images, fake_images, training):
        batch_size = tf.shape(real_images)[0]
        alpha = tf.random.uniform(shape=[batch_size], minval=0., maxval=1.)
        for _ in range(len(real_images.shape) - 1):
            alpha = tf.expand_dims(alpha, axis=-1)
        interpolate = real_images * alpha + fake_images * (1 - alpha)
        with tf.GradientTape() as tape:
            tape.watch(interpolate)
            prediction = self.discriminator(interpolate, training=training)
        grads = tape.gradient(prediction, interpolate)
        grads = tf.reshape(grads, shape=(batch_size, -1))
        grads = tf.square(tf.norm(grads, axis=-1) - 1)
        return tf.reduce_mean(grads, axis=0)

    @tf.function
    def _train_discriminator(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = self.prior.sample(batch_size)
        generated_images = self.generator(noise, training=True)
        with tf.GradientTape() as tape:
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            disc_loss = self._discriminator_loss(outputs=(real_output, fake_output))
            gp_loss = self._compute_gp(real_images, generated_images, training=True)
            disc_loss = disc_loss + gp_loss * self.gp_weight
        grads = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return {
            'disc_loss': disc_loss
        }

    def train(self,
              x=None,
              batch_size=None,
              epochs=1,
              callbacks=None):
        for callback in callbacks:
            callback.set_model(self)
        t = 0
        dataset = tf.data.Dataset.from_tensor_slices(x).shuffle(buffer_size=x.shape[0]).batch(batch_size)
        for i in range(1, epochs + 1):
            bar = tqdm(total=-(-x.shape[0] // batch_size))
            gen_loss = 0
            for images in dataset:
                disc_loss = self._train_discriminator(images)
                if (t == self.n_critics - 1):
                    gen_loss = self._train_generator(images)
                t = (t + 1) % self.n_critics
                bar.update(1)
                bar.set_description(f'Epoch {i}/{epochs}, disc_loss: {disc_loss:.4f}, gen_loss: {gen_loss:.4f}')
            bar.close()


class ACWassersteinGANGradientPenalty(ACGAN, WassersteinGANGradientPenalty):
    def _discriminator_loss(self, outputs):
        real_output, fake_output, real_logits, real_labels = outputs
        validity_loss = WassersteinGANGradientPenalty._discriminator_loss(self, outputs=(real_output, fake_output))
        real_class_loss = self._compute_classification_loss(real_logits, real_labels)
        loss = validity_loss + self.class_loss_weight * real_class_loss
        return loss

    def _compute_gp(self, real_images, fake_images, training):
        batch_size = tf.shape(real_images)[0]
        alpha = tf.random.uniform(shape=[batch_size], minval=0., maxval=1.)
        for _ in range(len(real_images.shape) - 1):
            alpha = tf.expand_dims(alpha, axis=-1)
        interpolate = real_images * alpha + fake_images * (1 - alpha)
        with tf.GradientTape() as tape:
            tape.watch(interpolate)
            validity, _ = self.discriminator(interpolate, training=training)  # the GP should only be in validity path
        grads = tape.gradient(validity, interpolate)
        grads = tf.reshape(grads, shape=(batch_size, -1))
        grads = tf.square(tf.norm(grads, axis=-1) - 1)
        return tf.reduce_mean(grads, axis=0)

    @tf.function
    def _train_discriminator(self, data):
        print('Tracing discriminator')
        real_images, real_labels = data
        batch_size = tf.shape(real_images)[0]
        noise = self.prior.sample(batch_size)
        generated_images = self.generator(inputs=(noise, real_labels), training=True)
        with tf.GradientTape() as tape:
            real_output, real_logits = self.discriminator(real_images, training=True)
            fake_output, _ = self.discriminator(generated_images, training=True)
            disc_loss = self._discriminator_loss(outputs=(real_output,
                                                          fake_output,
                                                          real_logits,
                                                          real_labels))
            gp_loss = self._compute_gp(real_images, generated_images, training=True)
            disc_loss = disc_loss + gp_loss * self.gp_weight
        grads = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        # compute accuracy
        accuracy = compute_accuracy(real_logits, real_labels)
        return disc_loss, accuracy

    def train(self,
              x=None,
              y=None,
              batch_size=None,
              epochs=1,
              callbacks=None):
        for callback in callbacks:
            callback.set_model(self)
        t = 0
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(buffer_size=x.shape[0]).batch(batch_size)
        for i in range(1, epochs + 1):
            disc_acc_metric = tf.keras.metrics.Mean()
            gen_acc_metric = tf.keras.metrics.Mean()
            disc_loss_metric = tf.keras.metrics.Mean()
            gen_loss_metric = tf.keras.metrics.Mean()
            bar = tqdm(total=-(-x.shape[0] // batch_size))
            for images, labels in dataset:
                disc_loss, disc_accuracy = self._train_discriminator(data=(images, labels))
                disc_loss_metric.update_state(disc_loss)
                disc_acc_metric.update_state(disc_accuracy)
                if (t == self.n_critics - 1):
                    gen_loss, gen_accuracy = self._train_generator(data=(images, labels))
                    gen_loss_metric.update_state(gen_loss)
                    gen_acc_metric.update_state(gen_accuracy)
                t = (t + 1) % self.n_critics
                bar.update(1)
                bar.set_description(f'Epoch {i}/{epochs}, disc_loss: {disc_loss_metric.result():.4f}, '
                                    f'gen_loss: {gen_loss_metric.result():.4f}, '
                                    f'disc_acc: {disc_acc_metric.result():.4f}, '
                                    f'gen_acc: {gen_acc_metric.result():.4f}')
            bar.close()

            for callback in callbacks:
                callback.on_epoch_end(i)
