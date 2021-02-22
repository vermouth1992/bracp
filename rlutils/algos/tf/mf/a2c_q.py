"""
Proximal Policy Optimization
"""

import time
from typing import Callable

import numpy as np
import tensorflow as tf
from rlutils.replay_buffers import PyUniformParallelEnvReplayBuffer
from rlutils.runner import TFRunner, run_func_as_main
from rlutils.tf.functional import to_numpy_or_python_type, hard_update, compute_target_value, soft_update
from rlutils.tf.nn import NormalActor, CategoricalActor, EnsembleMinQNet


class A2CQAgent(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, act_dtype, mlp_hidden=64,
                 pi_lr=1e-3, vf_lr=1e-3, clip_ratio=0.2,
                 entropy_coef=0.001, target_kl=0.05,
                 train_vf_iters=80
                 ):
        """
        Args:
            policy_net: The policy net must implement following methods:
                - forward: takes obs and return action_distribution and value
                - forward_action: takes obs and return action_distribution
                - forward_value: takes obs and return value.
            The advantage is that we can save computation if we only need to fetch parts of the graph. Also, we can
            implement policy and value in both shared and non-shared way.
            learning_rate:
            lam:
            clip_param:
            entropy_coef:
            target_kl:
            max_grad_norm:
        """
        super(A2CQAgent, self).__init__()
        if act_dtype == np.int32:
            self.policy_net = CategoricalActor(obs_dim=obs_dim, act_dim=act_dim, mlp_hidden=mlp_hidden)
        else:
            self.policy_net = NormalActor(obs_dim=obs_dim, act_dim=act_dim, mlp_hidden=mlp_hidden)
        self.pi_optimizer = tf.keras.optimizers.Adam(learning_rate=pi_lr)
        self.q_network = EnsembleMinQNet(ob_dim=obs_dim, ac_dim=act_dim, mlp_hidden=mlp_hidden)
        self.target_q_network = EnsembleMinQNet(ob_dim=obs_dim, ac_dim=act_dim, mlp_hidden=mlp_hidden)
        hard_update(self.target_q_network, self.q_network)
        self.q_optimizer = tf.keras.optimizers.Adam(learning_rate=vf_lr)

        self.target_kl = target_kl
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.train_vf_iters = train_vf_iters

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.logger = None

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        self.logger.log_tabular('PolicyLoss', average_only=True)
        self.logger.log_tabular('ValueLoss', average_only=True)
        self.logger.log_tabular('Entropy', average_only=True)

    def get_pi_distribution(self, obs, deterministic=tf.convert_to_tensor(False)):
        return self.policy_net((obs, deterministic))[-1]

    def call(self, inputs, training=None, mask=None):
        pi_distribution = self.get_pi_distribution(inputs)
        pi_action = pi_distribution.sample()
        return pi_action

    @tf.function
    def act_batch(self, obs):
        pi_distribution = self.get_pi_distribution(obs)
        pi_action = pi_distribution.sample()
        return pi_action

    @tf.function
    def _update_policy_step(self, obs, act):
        print(f'Tracing _update_policy_step with obs={obs}')
        with tf.GradientTape() as tape:
            distribution = self.get_pi_distribution(obs)
            entropy = tf.reduce_mean(distribution.entropy())
            log_prob = distribution.log_prob(act)
            adv = self.q_network((obs, act))
            policy_loss = -tf.reduce_mean(log_prob * adv)

            loss = policy_loss - entropy * self.entropy_coef

        gradients = tape.gradient(loss, self.policy_net.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(gradients, self.policy_net.trainable_variables))

        info = dict(
            PolicyLoss=policy_loss,
            Entropy=entropy,
        )
        return info

    @tf.function
    def _update_q(self, obs, act, rew, next_obs, done):
        next_q = self.target_q_network((next_obs, self.policy_net((next_obs, False))[0]))
        target_q = compute_target_value(rew, 0.99, done, next_q)
        with tf.GradientTape() as q_tape:
            q_values = self.q_network((obs, act), training=True)  # (num_ensembles, None)
            q_values_loss = 0.5 * tf.square(tf.expand_dims(target_q, axis=0) - q_values)
            # (num_ensembles, None)
            q_values_loss = tf.reduce_sum(q_values_loss, axis=0)  # (None,)
            # apply importance weights
            q_values_loss = tf.reduce_mean(q_values_loss)
        q_gradients = q_tape.gradient(q_values_loss, self.q_network.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_gradients, self.q_network.trainable_variables))

        soft_update(self.target_q_network, self.q_network, tau=1e-3)

        info = dict(
            ValueLoss=q_values_loss,
            VVals=q_values[0]
        )
        return info

    def update_policy(self, obs, act, next_obs, rew, done):
        assert tf.is_tensor(obs), f'obs must be a tf tensor. Got {obs}'

        info = self._update_policy_step(obs, act)
        # only record the final result
        info.update(self._update_q(obs, act, next_obs, rew, done))
        self.logger.store(**to_numpy_or_python_type(info))


class Runner(TFRunner):
    def setup_agent(self,
                    mlp_hidden,
                    pi_lr,
                    vf_lr,
                    clip_ratio,
                    entropy_coef,
                    target_kl,
                    train_vf_iters):
        # Instantiate policy
        obs_dim = self.env.single_observation_space.shape[0]
        act_dim = self.env.single_action_space.n if self.is_discrete_env else self.env.single_action_space.shape[0]
        act_dtype = np.int32 if self.is_discrete_env else np.float32
        self.agent = A2CQAgent(obs_dim=obs_dim, act_dim=act_dim, act_dtype=act_dtype, mlp_hidden=mlp_hidden,
                               pi_lr=pi_lr, vf_lr=vf_lr, clip_ratio=clip_ratio,
                               entropy_coef=entropy_coef, target_kl=target_kl,
                               train_vf_iters=train_vf_iters
                               )
        self.agent.set_logger(self.logger)

    def setup_replay_buffer(self,
                            max_length,
                            gamma,
                            lam):
        obs_dim = self.env.single_observation_space.shape[0]
        act_dim = self.env.single_action_space.shape[0]
        self.buffer = PyUniformParallelEnvReplayBuffer(obs_dim=obs_dim,
                                                       act_dim=act_dim,
                                                       act_dtype=np.float32,
                                                       capacity=max_length * self.num_parallel_env,
                                                       batch_size=None,
                                                       num_parallel_env=self.num_parallel_env)

    def run_one_step(self, t):
        """ Only collect dataset. No computation """
        act = self.agent.act_batch(tf.convert_to_tensor(self.obs, dtype=tf.float32))
        act = act.numpy()
        obs2, rew, dones, infos = self.env.step(act)
        time_truncated_dones = np.array([info.get('TimeLimit.truncated', False) for info in infos],
                                        dtype=np.bool_)
        true_d = np.logical_and(dones, not time_truncated_dones)
        self.buffer.add(data={
            'obs': self.obs,
            'act': act,
            'rew': rew,
            'next_obs': obs2,
            'done': true_d
        })
        self.ep_ret += rew
        self.ep_len += 1

        self.obs = obs2

    def on_train_begin(self):
        self.start_time = time.time()

    def on_epoch_begin(self, epoch):
        self.obs = self.env.reset()
        self.ep_ret = np.zeros(shape=self.num_parallel_env, dtype=np.float32)
        self.ep_len = np.zeros(shape=self.num_parallel_env, dtype=np.int32)

    def on_epoch_end(self, epoch):
        data = {k: tf.convert_to_tensor(v) for k, v in self.buffer.get().items()}
        self.agent.update_policy(**data)
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('VVals', with_min_and_max=True)
        self.logger.log_tabular('TotalEnvInteracts', epoch * self.steps_per_epoch * self.num_parallel_env)
        self.agent.log_tabular()
        self.logger.dump_tabular()

    @classmethod
    def main(cls, env_name, env_fn: Callable = None, mlp_hidden=256, seed=0, batch_size=5000, num_parallel_envs=5,
             epochs=200, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3,
             train_vf_iters=80, lam=0.97, max_ep_len=1000, target_kl=0.05, entropy_coef=1e-3, logger_path: str = None):
        # Instantiate environment
        assert batch_size % num_parallel_envs == 0

        steps_per_epoch = batch_size // num_parallel_envs

        config = locals()
        runner = cls(seed=seed, steps_per_epoch=steps_per_epoch,
                     epochs=epochs, exp_name=None, logger_path=logger_path)
        runner.setup_env(env_name=env_name, num_parallel_env=num_parallel_envs, env_fn=env_fn,
                         asynchronous=False, num_test_episodes=None)
        runner.setup_logger(config)
        runner.setup_agent(mlp_hidden=mlp_hidden,
                           pi_lr=pi_lr,
                           vf_lr=vf_lr,
                           clip_ratio=clip_ratio,
                           entropy_coef=entropy_coef,
                           target_kl=target_kl,
                           train_vf_iters=train_vf_iters)
        runner.setup_replay_buffer(max_length=steps_per_epoch, gamma=gamma, lam=lam)
        runner.run()


if __name__ == '__main__':
    run_func_as_main(Runner.main)
