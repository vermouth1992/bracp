import math

import tensorflow as tf


class _RandomGenerator(object):
    """Random generator that selects appropriate random ops."""
    dtypes = tf.dtypes

    def __init__(self, seed=None):
        super(_RandomGenerator, self).__init__()
        if seed is not None:
            # Stateless random ops requires 2-int seed.
            self.seed = [seed, 0]
        else:
            self.seed = None

    def random_normal(self, shape, mean=0.0, stddev=1, dtype=dtypes.float32):
        """A deterministic random normal if seed is passed."""
        if self.seed:
            op = tf.random.stateless_normal
        else:
            op = tf.random.normal
        return op(
            shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=self.seed)

    def random_uniform(self, shape, minval, maxval, dtype):
        """A deterministic random uniform if seed is passed."""
        if self.seed:
            op = tf.random.stateless_uniform
        else:
            op = tf.random.uniform
        return op(
            shape=shape, minval=minval, maxval=maxval, dtype=dtype, seed=self.seed)

    def truncated_normal(self, shape, mean, stddev, dtype):
        """A deterministic truncated normal if seed is passed."""
        if self.seed:
            op = tf.random.stateless_truncated_normal
        else:
            op = tf.random.truncated_normal
        return op(
            shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=self.seed)


class EnsembleVarianceScaling(tf.keras.initializers.Initializer):
    def __init__(self,
                 scale=1.0,
                 mode="fan_in",
                 distribution="truncated_normal",
                 seed=None):
        if scale <= 0.:
            raise ValueError("`scale` must be positive float.")
        if mode not in {"fan_in", "fan_out", "fan_avg"}:
            raise ValueError("Invalid `mode` argument:", mode)
        distribution = distribution.lower()
        # Compatibility with keras-team/keras.
        if distribution == "normal":
            distribution = "truncated_normal"
        if distribution not in {"uniform", "truncated_normal",
                                "untruncated_normal"}:
            raise ValueError("Invalid `distribution` argument:", distribution)
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed
        self._random_generator = _RandomGenerator(seed)

    def __call__(self, shape, dtype=tf.dtypes.float32):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. Only floating point types are
           supported.

        Raises:
          ValueError: If the dtype is not floating point
        """
        scale = self.scale
        scale_shape = shape
        fan_in, fan_out = _compute_fans(scale_shape)
        if self.mode == "fan_in":
            scale /= max(1., fan_in)
        elif self.mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        if self.distribution == "truncated_normal":
            # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale) / .87962566103423978
            return self._random_generator.truncated_normal(shape, 0.0, stddev, dtype)
        elif self.distribution == "untruncated_normal":
            stddev = math.sqrt(scale)
            return self._random_generator.random_normal(shape, 0.0, stddev, dtype)
        else:
            limit = math.sqrt(3.0 * scale)
            return self._random_generator.random_uniform(shape, -limit, limit, dtype)

    def get_config(self):
        return {
            "scale": self.scale,
            "mode": self.mode,
            "distribution": self.distribution,
            "seed": self.seed
        }


def _compute_fans(shape):
    """Computes the number of input and output units for a weight shape.

    Args:
      shape: Integer shape tuple or TF tensor shape.

    Returns:
      A tuple of integer scalars (fan_in, fan_out).
    """
    assert len(shape) == 3  # only used for ensemble dense layer
    fan_in = shape[1]
    fan_out = shape[2]
    return int(fan_in), int(fan_out)


class EnsembleHeNormal(EnsembleVarianceScaling):
    def __init__(self, seed=None):
        super(EnsembleHeNormal, self).__init__(
            scale=2., mode='fan_in', distribution='truncated_normal', seed=seed)

    def get_config(self):
        return {'seed': self.seed}


class EnsembleHeUniform(EnsembleVarianceScaling):
    def __init__(self, seed=None):
        super(EnsembleHeUniform, self).__init__(
            scale=2., mode='fan_in', distribution='uniform', seed=seed)

    def get_config(self):
        return {'seed': self.seed}


class EnsembleGlorotNormal(EnsembleVarianceScaling):
    def __init__(self, seed=None):
        super(EnsembleGlorotNormal, self).__init__(
            scale=1.0,
            mode='fan_avg',
            distribution='truncated_normal',
            seed=seed)

    def get_config(self):
        return {'seed': self.seed}


class EnsembleGlorotUniform(EnsembleVarianceScaling):
    def __init__(self, seed=None):
        super(EnsembleGlorotUniform, self).__init__(
            scale=1.0,
            mode='fan_avg',
            distribution='uniform',
            seed=seed)

    def get_config(self):
        return {'seed': self.seed}


ensemble_init = {
    'he_normal': EnsembleHeNormal,
    'he_uniform': EnsembleHeUniform,
    'glorot_normal': EnsembleGlorotNormal,
    'glorot_uniform': EnsembleGlorotUniform
}


def _decode_initializer(name):
    if name is None:
        name = 'glorot_uniform'
    if isinstance(name, str):
        if name in ensemble_init:
            return ensemble_init[name]
    return name
