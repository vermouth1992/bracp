import tensorflow as tf
import tensorflow_probability as tfp
from rlutils.tf.distributions import make_independent_normal_from_params, apply_squash_log_prob, \
    make_independent_centered_beta_from_params, make_independent_truncated_normal, make_independent_normal
from rlutils.tf.functional import clip_atanh

from .functional import build_mlp

tfd = tfp.distributions
LOG_STD_RANGE = (-20., 5.)
EPS = 1e-3


@tf.function
def get_pi_action(deterministic, pi_distribution):
    # print(f'Tracing get_pi_action with deterministic={deterministic}')
    return tf.cond(pred=deterministic, true_fn=lambda: pi_distribution.mean(),
                   false_fn=lambda: pi_distribution.sample())


@tf.function
def get_pi_action_categorical(deterministic, pi_distribution):
    # print(f'Tracing get_pi_action with deterministic={deterministic}')
    return tf.cond(pred=deterministic,
                   true_fn=lambda: tf.argmax(pi_distribution.probs_parameter(), axis=-1,
                                             output_type=pi_distribution.dtype),
                   false_fn=lambda: pi_distribution.sample())


class StochasticActor(tf.keras.Model):
    @property
    def pi_dist_layer(self):
        raise NotImplementedError

    def transform_raw_action(self, raw_actions):
        return raw_actions

    def inverse_transform_action(self, action):
        return action

    def transform_raw_log_prob(self, raw_log_prob, raw_action):
        return raw_log_prob


class CategoricalActor(StochasticActor):
    def __init__(self, obs_dim, act_dim, mlp_hidden):
        super(CategoricalActor, self).__init__()
        self.net = build_mlp(input_dim=obs_dim, output_dim=act_dim, mlp_hidden=mlp_hidden)

    @property
    def pi_dist_layer(self):
        return tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfd.Categorical(logits=t)
        )

    def call(self, inputs, **kwargs):
        inputs, deterministic = inputs
        params = self.net(inputs)
        pi_distribution = self.pi_dist_layer(params)
        pi_action = get_pi_action_categorical(deterministic, pi_distribution)
        logp_pi = pi_distribution.log_prob(pi_action)
        pi_action_final = pi_action
        return pi_action_final, logp_pi, pi_action, pi_distribution


class NormalActor(StochasticActor):
    def __init__(self, obs_dim, act_dim, mlp_hidden, global_std=True):
        super(NormalActor, self).__init__()
        self.global_std = global_std
        if self.global_std:
            self.net = build_mlp(input_dim=obs_dim, output_dim=act_dim, mlp_hidden=mlp_hidden)
            self.log_std = tf.Variable(initial_value=-0.5 * tf.ones(act_dim))
        else:
            self.net = build_mlp(input_dim=obs_dim, output_dim=act_dim * 2, mlp_hidden=mlp_hidden)
            self.log_std = None

    @property
    def pi_dist_layer(self):
        return tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: make_independent_normal(t[0], t[1]))

    def call(self, inputs, **kwargs):
        inputs, deterministic = inputs
        params = self.net(inputs)
        if self.global_std:
            pi_distribution = self.pi_dist_layer((params, tf.math.softplus(self.log_std)))
        else:
            mean, log_std = tf.split(params, 2, axis=-1)
            pi_distribution = self.pi_dist_layer((tf.tanh(mean), tf.math.softplus(log_std)))

        pi_action = get_pi_action(deterministic, pi_distribution)
        logp_pi = pi_distribution.log_prob(pi_action)
        pi_action_final = pi_action
        return pi_action_final, logp_pi, pi_action, pi_distribution


class TruncatedNormalActor(NormalActor):
    @property
    def pi_dist_layer(self):
        return tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: make_independent_truncated_normal(t[0], t[1]))


class CenteredBetaMLPActor(StochasticActor):
    """ Note that Beta distribution is 2x slower than SquashedGaussian"""

    def __init__(self, ob_dim, ac_dim, mlp_hidden):
        super(CenteredBetaMLPActor, self).__init__()
        self.net = build_mlp(ob_dim, ac_dim * 2, mlp_hidden)
        self.ac_dim = ac_dim
        self.build(input_shape=[(None, ob_dim), (None,)])

    @property
    def pi_dist_layer(self):
        return tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: make_independent_centered_beta_from_params(t))

    def call(self, inputs, **kwargs):
        inputs, deterministic = inputs
        # print(f'Tracing call with inputs={inputs}, deterministic={deterministic}')
        params = self.net(inputs)
        pi_distribution = self.pi_dist_layer(params)
        pi_action = get_pi_action(deterministic, pi_distribution)
        pi_action = tf.clip_by_value(pi_action, EPS, 1. - EPS)
        logp_pi = pi_distribution.log_prob(pi_action)
        return pi_action, logp_pi, pi_action, pi_distribution


class SquashedGaussianMLPActor(StochasticActor):
    def __init__(self, ob_dim, ac_dim, mlp_hidden):
        super(SquashedGaussianMLPActor, self).__init__()
        self.net = build_mlp(ob_dim, ac_dim * 2, mlp_hidden)
        self.ac_dim = ac_dim
        self.build(input_shape=[(None, ob_dim), (None,)])

    @property
    def pi_dist_layer(self):
        return tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: make_independent_normal_from_params(t, min_log_scale=LOG_STD_RANGE[0],
                                                                               max_log_scale=LOG_STD_RANGE[1]))

    def transform_raw_action(self, action):
        return tf.tanh(action)

    def inverse_transform_action(self, action):
        return clip_atanh(action)

    def transform_raw_log_prob(self, raw_log_prob, raw_action):
        return apply_squash_log_prob(raw_log_prob=raw_log_prob, x=raw_action)

    def call(self, inputs, **kwargs):
        inputs, deterministic = inputs
        params = self.net(inputs)
        pi_distribution = self.pi_dist_layer(params)
        pi_action = get_pi_action(deterministic, pi_distribution)
        logp_pi = pi_distribution.log_prob(pi_action)
        logp_pi = self.transform_raw_log_prob(logp_pi, pi_action)
        pi_action_final = self.transform_raw_action(pi_action)
        return pi_action_final, logp_pi, pi_action, pi_distribution
