"""
Implement soft actor critic agent here
"""

import rlutils.tf as rlu
import tensorflow as tf
from rlutils.runner import OffPolicyRunner, run_func_as_main, TFRunner


class SACAgent(tf.keras.Model):
    def __init__(self,
                 obs_spec,
                 act_spec,
                 policy_type='gaussian',
                 policy_mlp_hidden=128,
                 policy_lr=3e-4,
                 q_mlp_hidden=256,
                 q_lr=3e-4,
                 alpha=1.0,
                 alpha_lr=1e-3,
                 tau=5e-3,
                 gamma=0.99,
                 target_entropy=None,
                 ):
        super(SACAgent, self).__init__()
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.act_dim = self.act_spec.shape[0]
        if len(self.obs_spec.shape) == 1:  # 1D observation
            self.obs_dim = self.obs_spec.shape[0]
            if policy_type == 'gaussian':
                self.policy_net = rlu.nn.SquashedGaussianMLPActor(self.obs_dim, self.act_dim, policy_mlp_hidden)
            elif policy_type == 'beta':
                self.policy_net = rlu.nn.CenteredBetaMLPActor(self.obs_dim, self.act_dim, policy_mlp_hidden)
            else:
                raise NotImplementedError
            self.q_network = rlu.nn.EnsembleMinQNet(self.obs_dim, self.act_dim, q_mlp_hidden)
            self.target_q_network = rlu.nn.EnsembleMinQNet(self.obs_dim, self.act_dim, q_mlp_hidden)
        else:
            raise NotImplementedError
        rlu.functional.hard_update(self.target_q_network, self.q_network)

        self.policy_optimizer = tf.keras.optimizers.Adam(lr=policy_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(lr=q_lr)

        self.log_alpha = rlu.nn.LagrangeLayer(initial_value=alpha)
        self.alpha_optimizer = tf.keras.optimizers.Adam(lr=alpha_lr)
        self.target_entropy = -self.act_dim if target_entropy is None else target_entropy

        self.tau = tau
        self.gamma = gamma

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        for i in range(self.q_network.num_ensembles):
            self.logger.log_tabular(f'Q{i + 1}Vals', with_min_and_max=True)
        self.logger.log_tabular('LogPi', average_only=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('Alpha', average_only=True)
        self.logger.log_tabular('LossAlpha', average_only=True)

    @tf.function
    def update_target(self):
        rlu.functional.soft_update(self.target_q_network, self.q_network, self.tau)

    def _compute_next_obs_q(self, next_obs):
        alpha = self.log_alpha()
        next_action, next_action_log_prob, _, _ = self.policy_net((next_obs, False))
        next_q_values = self.target_q_network((next_obs, next_action), training=False) - alpha * next_action_log_prob
        return next_q_values

    @tf.function
    def _update_q_nets(self, obs, act, next_obs, done, rew):
        # compute target Q values
        next_q_values = self._compute_next_obs_q(next_obs)
        q_target = rlu.functional.compute_target_value(rew, self.gamma, done, next_q_values)

        # q loss
        with tf.GradientTape() as q_tape:
            q_values = self.q_network((obs, act), training=True)  # (num_ensembles, None)
            q_values_loss = 0.5 * tf.square(tf.expand_dims(q_target, axis=0) - q_values)
            # (num_ensembles, None)
            q_values_loss = tf.reduce_sum(q_values_loss, axis=0)  # (None,)
            # apply importance weights
            q_values_loss = tf.reduce_mean(q_values_loss)
        q_gradients = q_tape.gradient(q_values_loss, self.q_network.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_gradients, self.q_network.trainable_variables))

        info = dict(
            LossQ=q_values_loss,
        )
        for i in range(self.q_network.num_ensembles):
            info[f'Q{i + 1}Vals'] = q_values[i]
        return info

    @tf.function
    def _update_actor(self, obs):
        alpha = self.log_alpha()
        # policy loss
        with tf.GradientTape() as policy_tape:
            action, log_prob, _, _ = self.policy_net((obs, False))
            q_values_pi_min = self.q_network((obs, action), training=False)
            policy_loss = tf.reduce_mean(log_prob * alpha - q_values_pi_min)
        policy_gradients = policy_tape.gradient(policy_loss, self.policy_net.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_gradients, self.policy_net.trainable_variables))

        # log alpha
        with tf.GradientTape() as alpha_tape:
            alpha = self.log_alpha()
            alpha_loss = -tf.reduce_mean(alpha * (log_prob + self.target_entropy))
        alpha_gradient = alpha_tape.gradient(alpha_loss, self.log_alpha.trainable_variables)
        self.alpha_optimizer.apply_gradients(zip(alpha_gradient, self.log_alpha.trainable_variables))

        info = dict(
            LogPi=log_prob,
            Alpha=alpha,
            LossAlpha=alpha_loss,
            LossPi=policy_loss,
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
        print(f'Tracing train_step with {update_target}')
        info = self._update_q_nets(obs, act, next_obs, done, rew)
        if update_target:
            actor_info = self._update_actor(obs)
            info.update(actor_info)
            self.update_target()
        return info

    def train_on_batch(self, data, **kwargs):
        info = self.train_step(data=data)
        self.logger.store(**rlu.functional.to_numpy_or_python_type(info))

    @tf.function
    def act_batch(self, obs, deterministic):
        print(f'Tracing sac act_batch with obs {obs}')
        pi_final = self.policy_net((obs, deterministic))[0]
        return pi_final

    @tf.function
    def act_batch_test(self, obs):
        n = 20
        batch_size = tf.shape(obs)[0]
        obs = tf.tile(obs, (n, 1))
        action = self.policy_net((obs, False))[0]
        q_values_pi_min = self.q_network((obs, action), training=True)[0, :]
        action = tf.reshape(action, shape=(n, batch_size, self.act_dim))
        idx = tf.argmax(tf.reshape(q_values_pi_min, shape=(n, batch_size)), axis=0,
                        output_type=tf.int32)  # (batch_size)
        idx = tf.stack([idx, tf.range(batch_size)], axis=-1)
        pi_final = tf.gather_nd(action, idx)
        return pi_final


class Runner(OffPolicyRunner, TFRunner):
    def get_action_batch_explore(self, obs):
        return self.agent.act_batch(tf.convert_to_tensor(obs, tf.float32),
                                    tf.convert_to_tensor(False)).numpy()

    def get_action_batch_test(self, obs):
        return self.agent.act_batch(tf.convert_to_tensor(obs, tf.float32),
                                    tf.convert_to_tensor(True)).numpy()

    @classmethod
    def main(cls,
             env_name,
             # sac args
             nn_size=256,
             learning_rate=3e-4,
             alpha=0.2,
             tau=5e-3,
             gamma=0.99,
             **kwargs
             ):
        agent_kwargs = dict(
            policy_mlp_hidden=nn_size,
            policy_lr=learning_rate,
            q_mlp_hidden=nn_size,
            q_lr=learning_rate,
            alpha=alpha,
            alpha_lr=learning_rate,
            tau=tau,
            gamma=gamma,
            target_entropy=None
        )

        super(Runner, cls).main(
            env_name=env_name,
            agent_cls=SACAgent,
            agent_kwargs=agent_kwargs,
            **kwargs
        )


if __name__ == '__main__':
    run_func_as_main(Runner.main)
