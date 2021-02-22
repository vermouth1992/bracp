"""
Deep Q Network for low-dimensional observation space
"""

import rlutils.tf as rlu
import tensorflow as tf
from rlutils.runner import OffPolicyRunner, TFRunner, run_func_as_main


def gather_q_values(q_values, actions):
    batch_size = tf.shape(actions)[0]
    idx = tf.stack([tf.range(batch_size, dtype=actions.dtype), actions], axis=-1)  # (None, 2)
    q_values = tf.gather_nd(q_values, indices=idx)
    return q_values


class DQN(tf.keras.Model):
    def __init__(self,
                 obs_spec,
                 act_spec,
                 mlp_hidden=128,
                 double_q=True,
                 epsilon=0.1,
                 q_lr=1e-4,
                 gamma=0.99,
                 tau=5e-3,
                 huber_delta=None):
        super(DQN, self).__init__()
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        obs_dim = obs_spec.shape[0]
        act_dim = act_spec.n
        self.q_network = rlu.nn.build_mlp(obs_dim, act_dim, mlp_hidden=mlp_hidden, num_layers=3)
        self.target_q_network = rlu.nn.build_mlp(obs_dim, act_dim, mlp_hidden=mlp_hidden, num_layers=3)
        self.q_optimizer = tf.keras.optimizers.Adam(lr=q_lr)
        self.epsilon = tf.Variable(initial_value=epsilon, dtype=tf.float32, trainable=False)
        self.act_dim = act_dim
        self.double_q = double_q
        self.huber_delta = huber_delta
        self.gamma = gamma
        self.tau = tau
        reduction = tf.keras.losses.Reduction.NONE  # Note: tensorflow uses reduce_mean at axis=-1 by default
        if huber_delta is None:
            self.loss_fn = tf.keras.losses.MeanSquaredError(reduction=reduction)
        else:
            self.loss_fn = tf.keras.losses.Huber(delta=huber_delta, reduction=reduction)
        rlu.functional.hard_update(self.target_q_network, self.q_network)

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        self.logger.log_tabular('QVals', with_min_and_max=True)
        self.logger.log_tabular('LossQ', average_only=True)

    def set_epsilon(self, epsilon):
        assert epsilon >= 0. and epsilon <= 1.
        self.epsilon.assign(epsilon)

    @tf.function
    def update_target(self):
        rlu.functional.soft_update(self.target_q_network, self.q_network, tau=self.tau)

    @tf.function
    def _update_nets(self, obs, act, next_obs, done, rew):
        print('Tracing _update_nets')
        # compute target Q values
        target_q_values = self.target_q_network(next_obs)
        if self.double_q:
            # select action using Q network instead of target Q network
            target_actions = tf.argmax(self.q_network(next_obs), axis=-1, output_type=self.act_spec.dtype)
            target_q_values = gather_q_values(target_q_values, target_actions)
        else:
            target_q_values = tf.reduce_max(target_q_values, axis=-1)
        target_q_values = rew + self.gamma * (1. - done) * target_q_values
        with tf.GradientTape() as tape:
            q_values = gather_q_values(self.q_network(obs), act)  # (None,)
            loss = self.loss_fn(q_values, target_q_values)  # (None,)
        grad = tape.gradient(loss, self.q_network.trainable_variables)
        self.q_optimizer.apply_gradients(zip(grad, self.q_network.trainable_variables))
        info = dict(
            QVals=q_values,
            LossQ=loss
        )
        return info

    @tf.function
    def train_step(self, data):
        obs = data['obs']
        act = data['act']
        next_obs = data['next_obs']
        done = data['done']
        rew = data['rew']
        update_target = data['update_target']
        info = self._update_nets(obs, act, next_obs, done, rew)
        if update_target:
            self.update_target()
        return info

    def train_on_batch(self, data, **kwargs):
        info = self.train_step(data=data)
        self.logger.store(**rlu.functional.to_numpy_or_python_type(info))

    @tf.function
    def act_batch(self, obs, deterministic):
        """ Implement epsilon-greedy here """
        batch_size = tf.shape(obs)[0]
        epsilon = tf.random.uniform(shape=(batch_size,), minval=0., maxval=1., dtype=tf.float32)
        epsilon_indicator = tf.cast(epsilon > self.epsilon, dtype=tf.int32)  # (None,)
        random_actions = tf.random.uniform(shape=(batch_size,), minval=0, maxval=self.act_dim,
                                           dtype=self.act_spec.dtype)
        deterministic_actions = tf.argmax(self.q_network(obs), axis=-1, output_type=self.act_spec.dtype)
        epsilon_greedy_actions = tf.stack([random_actions, deterministic_actions], axis=-1)  # (None, 2)
        epsilon_greedy_actions = gather_q_values(epsilon_greedy_actions, epsilon_indicator)
        final_actions = tf.cond(deterministic, true_fn=lambda: deterministic_actions,
                                false_fn=lambda: epsilon_greedy_actions)
        return final_actions


class Runner(OffPolicyRunner, TFRunner):
    def get_action_batch_test(self, obs):
        return self.agent.act_batch(tf.convert_to_tensor(obs, dtype=tf.float32),
                                    tf.convert_to_tensor(True, dtype=tf.bool)).numpy()

    def get_action_batch_explore(self, obs):
        return self.agent.act_batch(self.o, deterministic=tf.convert_to_tensor(False)).numpy()

    @classmethod
    def main(cls,
             env_name,
             mlp_hidden=256,
             double_q=True,
             q_lr=1e-4,
             gamma=0.99,
             huber_delta: float = None,
             tau=5e-3,
             epsilon=0.1,
             **kwargs
             ):
        agent_kwargs = dict(
            mlp_hidden=mlp_hidden,
            double_q=double_q,
            q_lr=q_lr,
            gamma=gamma,
            huber_delta=huber_delta,
            tau=tau,
            epsilon=epsilon
        )

        super(Runner, cls).main(env_name=env_name,
                                agent_cls=DQN,
                                agent_kwargs=agent_kwargs,
                                **kwargs
                                )


if __name__ == '__main__':
    run_func_as_main(Runner.main)
