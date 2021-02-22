"""
Implementing dynamics model to perform Model-based RL
"""

import tensorflow as tf
import tensorflow_probability as tfp
from rlutils.future.optimizer import get_adam_optimizer, minimize
from rlutils.tf.callbacks import EpochLoggerCallback
from rlutils.tf.distributions import make_independent_normal_from_params
from rlutils.tf.functional import expand_ensemble_dim
from rlutils.tf.preprocessing import StandardScaler

from .functional import build_mlp

tfl = tfp.layers


class DynamicsModelRNNCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, model, reward_fn, terminate_fn, obs_normalizer, act_normalizer, delta_obs_normalizer,
                 rew_normalizer, num_particles, num_ensembles, obs_dim, act_dim, policy=None):
        super(DynamicsModelRNNCell, self).__init__()
        self.policy = policy
        self.model = model
        self.reward_fn = reward_fn
        self.terminate_fn = terminate_fn
        self.obs_normalizer = obs_normalizer
        self.act_normalizer = act_normalizer
        self.delta_obs_normalizer = delta_obs_normalizer
        self.rew_normalizer = rew_normalizer
        self.num_particles = num_particles
        self.num_ensembles = num_ensembles
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    @property
    def state_size(self):
        return tf.TensorShape([self.num_particles, self.obs_dim])

    @property
    def output_size(self):
        return tf.TensorShape([self.num_particles, self.obs_dim + 2])

    def sample_bootstrap(self, batch_size):
        return tf.stack(
            [tf.random.uniform(shape=(self.num_particles * batch_size,), maxval=self.num_ensembles, dtype=tf.int32),
             tf.range(self.num_particles * batch_size)], axis=-1)

    def call(self, inputs, states):
        states, = states
        batch_size = tf.shape(states)[0]
        states = tf.reshape(states, shape=(batch_size * self.num_particles, self.obs_dim))  # (None * P, obs_dim)

        if self.policy is None:
            current_action = tf.tile(tf.expand_dims(inputs, axis=1), (1, self.num_particles, 1))  # (None, P, ac_dim)
            current_action = tf.reshape(current_action, shape=(batch_size * self.num_particles, self.act_dim))
        else:
            current_action = self.policy(states)

        current_action = self.act_normalizer(current_action)
        state_norm = self.obs_normalizer(states)

        inputs = tf.concat((state_norm, current_action), axis=-1)
        output = self.model(inputs=inputs, training=False)
        output = output.sample()  # (num_ensemble, None * P, ob_dim)

        # for each particle, sample one bootstrap
        bootstrap_idx = self.sample_bootstrap(batch_size=batch_size)
        # for each particle, each sample, select one bootstrap
        output = tf.gather_nd(output, bootstrap_idx)  # (None * P, ob_dim)

        if self.reward_fn is None:
            delta_state_norm = output[:, :-1]
            reward_norm = output[:, -1]
            delta_state = self.delta_obs_normalizer.inverse_call(delta_state_norm)
            reward = self.rew_normalizer.inverse_call(reward_norm)  # (P * None)
            next_states = delta_state + states  # (None * P, ob_dim)
        else:
            delta_state_norm = output
            delta_state = self.delta_obs_normalizer.inverse_call(delta_state_norm)
            next_states = delta_state + states  # (P * None, ob_dim)
            reward = self.reward_fn(states, current_action, next_states)  # (None * P)

        if self.terminate_fn is not None:
            done = tf.cast(self.terminate_fn(states, current_action, next_states), dtype=tf.float32)
            done = tf.reshape(done, shape=(batch_size, self.num_particles, 1))
        else:
            done = tf.zeros(shape=[batch_size, self.num_particles, 1], dtype=tf.float32)

        next_states = tf.reshape(next_states, shape=(batch_size, self.num_particles, self.obs_dim))
        reward = tf.reshape(reward, shape=(batch_size, self.num_particles, 1))

        output = tf.concat((next_states, reward, done), axis=-1)
        return output, [next_states]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        raise NotImplementedError


class EnsembleDynamicsModel(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, mlp_hidden=64, num_ensembles=5, lr=1e-3,
                 reward_fn=None, terminate_fn=None):
        super(EnsembleDynamicsModel, self).__init__()
        self.lr = lr
        self.mlp_hidden = mlp_hidden
        self.num_ensembles = num_ensembles
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.reward_fn = reward_fn
        self.terminate_fn = terminate_fn
        self.obs_normalizer = StandardScaler(input_shape=[None, self.obs_dim])
        self.action_normalizer = StandardScaler(input_shape=[None, self.act_dim])
        self.delta_obs_normalizer = StandardScaler(input_shape=[None, self.obs_dim])
        if self.reward_fn is None:
            self.rew_normalizer = StandardScaler(input_shape=[None])
        else:
            self.rew_normalizer = None
        self.logger = None
        self.model = self.build_computation_graph()
        self.compile(optimizer=get_adam_optimizer(lr=self.lr))

    def set_statistics(self, obs, act, next_obs, rew):
        self.obs_normalizer.adapt(data=obs)
        self.action_normalizer.adapt(data=act)
        self.delta_obs_normalizer.adapt(data=next_obs - obs)
        if self.reward_fn is None:
            self.rew_normalizer.adapt(data=rew)

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        self.logger.log_tabular('TrainModelLoss', average_only=True)
        self.logger.log_tabular('ValModelLoss', average_only=True)

    def build_computation_graph(self):
        """
        The input is always without ensemble. It will output (ensemble, None, ...)
        """

        if self.reward_fn is None:
            output_dim = (self.obs_dim + 1) * 2
        else:
            output_dim = self.obs_dim * 2

        inputs_ph = tf.keras.Input(shape=(self.obs_dim + self.act_dim))
        inputs = tf.tile(tf.expand_dims(inputs_ph, axis=0), (self.num_ensembles, 1, 1))
        mlp = build_mlp(self.obs_dim + self.act_dim, output_dim, mlp_hidden=self.mlp_hidden,
                        num_ensembles=self.num_ensembles, squeeze=False, layer_norm=True, num_layers=3,
                        activation='relu')
        mlp.add(tfl.DistributionLambda(make_distribution_fn=lambda t: make_independent_normal_from_params(t)))
        outputs = mlp(inputs)
        model = tf.keras.Model(inputs=inputs_ph, outputs=outputs)
        return model

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data=data)
        obs, act = x
        next_obs, rew = y
        delta_obs = next_obs - obs
        norm_obs = self.obs_normalizer(obs)
        norm_act = self.action_normalizer(act)
        inputs = tf.concat((norm_obs, norm_act), axis=-1)
        norm_delta_obs = self.delta_obs_normalizer(delta_obs)  # (None, obs_dim)
        if self.reward_fn is None:
            outputs = tf.concat((norm_delta_obs, tf.expand_dims(rew, axis=-1)), axis=-1)
        else:
            outputs = norm_delta_obs

        outputs = expand_ensemble_dim(outputs, num_ensembles=self.num_ensembles)
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            out_dist = self.model(inputs, training=True)  # (ensemble, None)
            nll = -out_dist.log_prob(outputs)
            nll = tf.reduce_sum(nll, axis=0)
            loss = tf.reduce_mean(nll, axis=0)

        minimize(loss, tape, self.model, optimizer=self.optimizer)

        return {
            'loss': loss / self.num_ensembles
        }

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data=data)
        obs, act = x
        next_obs, rew = y
        delta_obs = next_obs - obs
        norm_obs = self.obs_normalizer(obs)
        norm_act = self.action_normalizer(act)
        inputs = tf.concat((norm_obs, norm_act), axis=-1)
        norm_delta_obs = self.delta_obs_normalizer(delta_obs)  # (None, obs_dim)
        if self.reward_fn is None:
            outputs = tf.concat((norm_delta_obs, tf.expand_dims(rew, axis=-1)), axis=-1)
        else:
            outputs = norm_delta_obs

        outputs = expand_ensemble_dim(outputs, num_ensembles=self.num_ensembles)
        out_dist = self.model(inputs, training=False)
        nll = -out_dist.log_prob(outputs)  # (ensemble, None)
        nll = tf.reduce_sum(nll, axis=0)
        loss = tf.reduce_mean(nll, axis=0)

        return {
            'loss': loss / self.num_ensembles
        }

    def update(self, inputs, sample_weights=None, batch_size=64, num_epochs=60, patience=None,
               validation_split=0.1, shuffle=True):
        obs = inputs['obs']
        actions = inputs['act']
        next_obs = inputs['next_obs']
        rewards = inputs['rew']
        # get data statistic
        self.set_statistics(obs, actions, next_obs, rewards)

        callbacks = [EpochLoggerCallback(keys=[('TrainModelLoss', 'loss'), ('ValModelLoss', 'val_loss')],
                                         epochs=num_epochs, logger=self.logger)]

        if patience is not None:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                                              restore_best_weights=True))

        self.fit(x=(obs, actions), y=(next_obs, rewards), sample_weight=sample_weights, epochs=num_epochs,
                 batch_size=batch_size, verbose=False, validation_split=validation_split,
                 callbacks=callbacks, shuffle=shuffle)

    def build_ts_model(self, horizon, num_particles, policy=None):
        """ Given obs and a sequence of actions, generate a sequence of next_obs, rewards and dones. """
        cell = DynamicsModelRNNCell(model=self.model, reward_fn=self.reward_fn, terminate_fn=self.terminate_fn,
                                    obs_normalizer=self.obs_normalizer, act_normalizer=self.action_normalizer,
                                    delta_obs_normalizer=self.delta_obs_normalizer, rew_normalizer=self.rew_normalizer,
                                    num_particles=num_particles, num_ensembles=self.num_ensembles, obs_dim=self.obs_dim,
                                    act_dim=self.act_dim, policy=policy)
        self.rnn = tf.keras.layers.RNN(cell, return_sequences=True, unroll=True)
        initial_states_ph = tf.keras.Input(shape=(num_particles, self.obs_dim))
        action_seq_ph = tf.keras.Input(shape=(horizon, self.act_dim))

        outputs = self.rnn(inputs=action_seq_ph, initial_state=initial_states_ph)

        next_obs = outputs[:, :, :, :self.obs_dim]
        rewards = outputs[:, :, :, self.obs_dim]
        dones = outputs[:, :, :, self.obs_dim + 1]

        model = tf.keras.Model(inputs=[initial_states_ph, action_seq_ph],
                               outputs=[next_obs, rewards, dones])
        model.sample_action = lambda shape: tf.random.uniform(shape=list(shape) + [self.act_dim],
                                                              minval=-1., maxval=1.,
                                                              dtype=tf.float32)
        model.num_particles = num_particles
        return model

    def call(self, inputs, training=None, mask=None):
        """
        Args:
            state: (None, ob_dim)
            action: (None, ac_dim)
        Returns: (None, ob_dim), (None), (None)
        """
        state, action, sample = inputs
        batch_size = tf.shape(state)[0]
        print(f'Tracing _predict_obs with state={state}, action={action}')
        norm_state = self.state_normalizer(state)
        norm_action = self.action_normalizer(action)
        sa = tf.concat((norm_state, norm_action), axis=-1)
        output = self.model(inputs=(sa,), training=training)[0]  # (num_ensembles, None, ob_dim)

        if sample:
            output = output.sample()
            # randomly select one bootstrap for each data
            bootstrap_idx = tf.stack(
                [tf.random.uniform(shape=(batch_size,), maxval=self.num_ensembles, dtype=tf.int32),
                 tf.range(batch_size)], axis=-1)
            output = tf.gather_nd(output, bootstrap_idx)  # (None, ob_dim)
        else:
            output = tf.reduce_mean(output.mean(), axis=0)

        if self.reward_fn is None:
            delta_state_norm = output[:, :-1]
            reward_norm = output[:, -1]
            reward = self.reward_normalizer.inverse_call(reward_norm)
            delta_state = self.delta_state_normalizer.inverse_call(delta_state_norm)
            next_state = delta_state + state
        else:
            print('Using external reward function')
            delta_state_norm = output
            delta_state = self.delta_state_normalizer.inverse_call(delta_state_norm)
            next_state = delta_state + state
            reward = self.reward_fn(state, action, next_state)

        if self.terminate_fn is None:
            dones = tf.zeros(shape=[batch_size], dtype=tf.bool)
        else:
            print('Using external terminate function')
            dones = self.terminate_fn(state, action, next_state)

        return next_state, reward, dones

    @tf.function
    def predict_obs(self, state, action, sample=True):
        """
        Args:
            state: (None, ob_dim)
            action: (None, ac_dim)
        Returns: (None, ob_dim), (None), (None,)
        """
        return self(inputs=(state, action, sample), training=False)
