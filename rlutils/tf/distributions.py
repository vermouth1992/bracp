import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
tfl = tfp.layers

EPS = 1e-4


class CenteredBeta(tfd.TransformedDistribution):
    def __init__(self,
                 concentration1,
                 concentration0,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='CenteredBeta'):
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            super(CenteredBeta, self).__init__(
                distribution=tfd.Beta(concentration1=concentration1, concentration0=concentration0),
                bijector=tfb.Chain(bijectors=[
                    tfb.Shift(shift=-1.),
                    tfb.Scale(scale=2.)
                ]),
                validate_args=validate_args,
                parameters=parameters,
                name=name)


def apply_squash_log_prob(raw_log_prob, x):
    """ Compute the log probability after applying tanh on raw_actions
    Args:
        raw_log_prob: (None,)
        raw_actions: (None, act_dim)
    Returns:
    """
    log_det_jacobian = 2. * (np.log(2.) - x - tf.math.softplus(-2. * x))
    num_reduce_dim = tf.rank(x) - tf.rank(raw_log_prob)
    log_det_jacobian = tf.reduce_sum(log_det_jacobian, axis=tf.range(-num_reduce_dim, 0))
    log_prob = raw_log_prob - log_det_jacobian
    return log_prob


def make_independent_normal(loc, scale, ndims=1):
    distribution = tfd.Independent(distribution=tfd.Normal(loc=loc, scale=scale),
                                   reinterpreted_batch_ndims=ndims)
    return distribution


def make_independent_normal_from_params(params, ndims=1, min_log_scale=None, max_log_scale=None):
    loc_params, scale_params = tf.split(params, 2, axis=-1)
    if min_log_scale is not None:
        scale_params = tf.maximum(scale_params, min_log_scale)
    if max_log_scale is not None:
        scale_params = tf.minimum(scale_params, max_log_scale)
    scale_params = tf.math.softplus(scale_params)
    distribution = make_independent_normal(loc_params, scale_params, ndims=ndims)
    return distribution


def make_independent_truncated_normal(loc, scale, low, high, ndims=1):
    pi_distribution = tfd.Independent(distribution=tfd.TruncatedNormal(loc=loc, scale=scale,
                                                                       low=low, high=high),
                                      reinterpreted_batch_ndims=ndims)
    return pi_distribution


def make_independent_truncated_normal_from_params(params, low, high, ndims=1, min_log_scale=None, max_log_scale=None):
    loc_params, scale_params = tf.split(params, 2, axis=-1)
    if min_log_scale is not None:
        scale_params = tf.maximum(scale_params, min_log_scale)
    if max_log_scale is not None:
        scale_params = tf.minimum(scale_params, max_log_scale)
    scale_params = tf.math.softplus(scale_params)
    distribution = make_independent_truncated_normal(loc_params, scale_params, low, high, ndims=ndims)
    return distribution


def make_independent_centered_beta(c1, c2, ndims=1):
    beta_distribution = CenteredBeta(concentration1=c1, concentration0=c2, validate_args=False, allow_nan_stats=False)
    distribution = tfd.Independent(beta_distribution,
                                   reinterpreted_batch_ndims=ndims)
    return distribution


def make_independent_centered_beta_from_params(params, ndims=1):
    params = tf.math.softplus(params) + 1.0
    c1, c2 = tf.split(params, 2, axis=-1)
    distribution = make_independent_centered_beta(c1, c2, ndims=ndims)
    return distribution


def make_independent_beta(c1, c2, ndims=1):
    beta_distribution = tfd.Beta(concentration1=c1, concentration0=c2, validate_args=False, allow_nan_stats=False)
    distribution = tfd.Independent(beta_distribution,
                                   reinterpreted_batch_ndims=ndims)
    return distribution


def make_independent_beta_from_params(params, ndims=1):
    params = tf.math.softplus(params) + 1.0
    c1, c2 = tf.split(params, 2, axis=-1)
    distribution = make_independent_beta(c1, c2, ndims=ndims)
    return distribution


class IndependentNormal(tfl.DistributionLambda):
    def __init__(self, min_log_scale=None, max_log_scale=None, ndims=1):
        super(IndependentNormal, self).__init__(make_distribution_fn=lambda t: make_independent_normal_from_params(
            t, ndims=ndims, min_log_scale=min_log_scale, max_log_scale=max_log_scale))


class IndependentBeta(tfl.DistributionLambda):
    def __init__(self, ndims=1):
        super(IndependentBeta, self).__init__(make_distribution_fn=lambda t: make_independent_beta_from_params(
            t, ndims=ndims))


class IndependentTruncatedNormal(tfl.DistributionLambda):
    def __init__(self, low, high, min_log_scale=None, max_log_scale=None, ndims=1):
        super(IndependentTruncatedNormal, self).__init__(
            make_distribution_fn=lambda t: make_independent_truncated_normal_from_params(
                t, low=low, high=high, ndims=ndims, min_log_scale=min_log_scale, max_log_scale=max_log_scale))


class IndependentCenteredBeta(tfl.DistributionLambda):
    def __init__(self, ndims=1):
        super(IndependentCenteredBeta, self).__init__(lambda t: make_independent_centered_beta_from_params(
            t, ndims=ndims))
