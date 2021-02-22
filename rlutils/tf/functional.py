import numpy as np
import tensorflow as tf

EPS = 1e-6


def compute_accuracy(logits, labels):
    num = tf.cast(tf.argmax(logits, axis=-1, output_type=tf.int32) == labels, dtype=tf.float32)
    accuracy = tf.reduce_mean(num)
    return accuracy


def expand_ensemble_dim(x, num_ensembles):
    """ functionality for outer class to expand before passing into the ensemble model. """
    multiples = tf.concat(([num_ensembles], tf.ones_like(tf.shape(x))), axis=0)
    x = tf.tile(tf.expand_dims(x, axis=0), multiples=multiples)
    return x


def clip_by_value(t, clip_value_min=None, clip_value_max=None):
    if clip_value_min is not None:
        t = tf.maximum(t, clip_value_min)
    if clip_value_max is not None:
        t = tf.minimum(t, clip_value_max)
    return t


def flatten_leading_dims(tensor, n_dims):
    if n_dims <= 1:
        return tensor
    newshape = [tf.math.reduce_prod(tf.shape(tensor)[:n_dims])] + tf.TensorShape(tf.shape(tensor)[n_dims:])
    return tf.reshape(tensor, shape=newshape)


def clip_atanh(x, name=None):
    return tf.atanh(tf.clip_by_value(x, clip_value_min=-1. + EPS, clip_value_max=1. - EPS), name=name)


@tf.function
def compute_target_value(reward, gamma, done, next_q):
    q_target = reward + gamma * (1.0 - done) * next_q
    return q_target


@tf.function
def flat_grads(grads):
    print(f'Tracing flat_grads grads={len(grads)}')
    grads = [tf.reshape(grad, shape=(-1,)) for grad in grads]
    return tf.concat(grads, axis=0)


@tf.function
def get_flat_trainable_variables(model: tf.keras.layers.Layer):
    print(f'Tracing get_flat_params_from model={model.name}')
    trainable_variables = [tf.reshape(p, shape=(-1,)) for p in model.trainable_variables]
    trainable_variables = tf.concat(trainable_variables, axis=0)
    return trainable_variables


@tf.function
def set_flat_trainable_variables(model: tf.keras.layers.Layer, trainable_variables):
    print(f'Tracing set_flat_params_to model={model.name}, flat_params={len(trainable_variables)}')
    prev_ind = 0
    for param in model.trainable_variables:
        flat_size = tf.reduce_prod(param.shape)
        param.assign(tf.reshape(trainable_variables[prev_ind:prev_ind + flat_size], shape=param.shape))
        prev_ind += flat_size


def soft_update(target: tf.keras.layers.Layer, source: tf.keras.layers.Layer, tau):
    print('Tracing soft_update_tf')
    for target_param, source_param in zip(target.variables, source.variables):
        target_param.assign(target_param * (1. - tau) + source_param * tau)


def hard_update(target: tf.keras.layers.Layer, source: tf.keras.layers.Layer):
    print('Tracing hard_update_tf')
    for target_param, source_param in zip(target.variables, source.variables):
        target_param.assign(source_param)


def to_numpy_or_python_type(tensors):
    """Converts a structure of `Tensor`s to `NumPy` arrays or Python scalar types.

    For each tensor, it calls `tensor.numpy()`. If the result is a scalar value,
    it converts it to a Python type, such as a float or int, by calling
    `result.item()`.

    Numpy scalars are converted, as Python types are often more convenient to deal
    with. This is especially useful for bfloat16 Numpy scalars, which don't
    support as many operations as other Numpy values.

    Args:
      tensors: A structure of tensors.

    Returns:
      `tensors`, but scalar tensors are converted to Python types and non-scalar
      tensors are converted to Numpy arrays.
    """

    def _to_single_numpy_or_python_type(t):
        if isinstance(t, tf.Tensor):
            x = t.numpy()
            return x.item() if np.ndim(x) == 0 else x
        return t  # Don't turn ragged or sparse tensors to NumPy.

    return tf.nest.map_structure(_to_single_numpy_or_python_type, tensors)
