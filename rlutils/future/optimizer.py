import tensorflow as tf


def get_adam_optimizer(lr, **kwargs):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        **kwargs
    )
    _ = optimizer.iterations  # this access will invoke optimizer._iterations method and create optimizer.iter attribute
    _ = optimizer.beta_1
    _ = optimizer.beta_2
    _ = optimizer.decay
    return optimizer


def minimize(loss, tape, model, optimizer=None):
    grads = tape.gradient(loss, model.trainable_variables)
    if optimizer is None:
        optimizer = model.optimizer
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return grads
