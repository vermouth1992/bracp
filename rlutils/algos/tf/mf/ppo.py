"""
Proximal Policy Optimization
"""

import time
from typing import Callable

import numpy as np
import tensorflow as tf
from rlutils.replay_buffers import GAEBuffer
from rlutils.runner import TFRunner, run_func_as_main
from rlutils.tf.functional import to_numpy_or_python_type
from rlutils.tf.nn import CategoricalActor, CenteredBetaMLPActor
from rlutils.tf.nn.functional import build_mlp


class PPOAgent(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, act_dtype, mlp_hidden=64,
                 pi_lr=1e-3, vf_lr=1e-3, clip_ratio=0.2,
                 entropy_coef=0.001, target_kl=0.05,
                 train_pi_iters=80, train_vf_iters=80
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
        super(PPOAgent, self).__init__()
        if act_dtype == np.int32:
            self.policy_net = CategoricalActor(obs_dim=obs_dim, act_dim=act_dim, mlp_hidden=mlp_hidden)
        else:
            self.policy_net = CenteredBetaMLPActor(obs_dim, act_dim, mlp_hidden=mlp_hidden)
        self.pi_optimizer = tf.keras.optimizers.Adam(learning_rate=pi_lr)
        self.v_optimizer = tf.keras.optimizers.Adam(learning_rate=vf_lr)
        self.value_net = build_mlp(input_dim=obs_dim, output_dim=1, squeeze=True, mlp_hidden=mlp_hidden)
        self.value_net.compile(optimizer=self.v_optimizer, loss='mse')

        self.target_kl = target_kl
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.train_pi_iters = train_pi_iters
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
        self.logger.log_tabular('AvgKL', average_only=True)
        self.logger.log_tabular('StopIter', average_only=True)

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
        log_prob = pi_distribution.log_prob(pi_action)
        v = self.value_net(obs)
        return pi_action, log_prob, v

    @tf.function
    def _update_policy_step(self, obs, act, adv, old_log_prob):
        print(f'Tracing _update_policy_step with obs={obs}')
        with tf.GradientTape() as tape:
            distribution = self.get_pi_distribution(obs)
            entropy = tf.reduce_mean(distribution.entropy())
            log_prob = distribution.log_prob(act)
            negative_approx_kl = log_prob - old_log_prob
            approx_kl_mean = tf.reduce_mean(-negative_approx_kl)

            ratio = tf.exp(negative_approx_kl)
            surr1 = ratio * adv
            surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            loss = policy_loss - entropy * self.entropy_coef

        gradients = tape.gradient(loss, self.policy_net.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(gradients, self.policy_net.trainable_variables))

        info = dict(
            PolicyLoss=policy_loss,
            Entropy=entropy,
            AvgKL=approx_kl_mean,
        )
        return info

    def update_policy(self, obs, act, ret, adv, logp):
        assert tf.is_tensor(obs), f'obs must be a tf tensor. Got {obs}'
        for i in range(self.train_pi_iters):
            info = self._update_policy_step(obs, act, adv, logp)
            if info['AvgKL'] > 1.5 * self.target_kl:
                self.logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break

        self.logger.store(StopIter=i)

        for i in range(self.train_vf_iters):
            loss = self.value_net.train_on_batch(x=obs, y=ret)

        # only record the final result
        info['ValueLoss'] = loss
        self.logger.store(**to_numpy_or_python_type(info))


class Runner(TFRunner):
    def setup_agent(self,
                    mlp_hidden,
                    pi_lr,
                    vf_lr,
                    clip_ratio,
                    entropy_coef,
                    target_kl,
                    train_pi_iters,
                    train_vf_iters):
        # Instantiate policy
        obs_dim = self.env.single_observation_space.shape[0]
        act_dim = self.env.single_action_space.n if self.is_discrete_env else self.env.single_action_space.shape[0]
        act_dtype = np.int32 if self.is_discrete_env else np.float32
        self.agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, act_dtype=act_dtype, mlp_hidden=mlp_hidden,
                              pi_lr=pi_lr, vf_lr=vf_lr, clip_ratio=clip_ratio,
                              entropy_coef=entropy_coef, target_kl=target_kl,
                              train_pi_iters=train_pi_iters, train_vf_iters=train_vf_iters
                              )
        self.agent.set_logger(self.logger)

    def setup_replay_buffer(self,
                            max_length,
                            gamma,
                            lam):
        obs_shape = self.env.single_observation_space.shape
        obs_dtype = self.env.single_observation_space.dtype
        if obs_dtype == np.float64:
            obs_dtype = np.float32
        act_shape = self.env.single_action_space.shape
        act_dtype = np.int32 if self.is_discrete_env else np.float32
        self.buffer = GAEBuffer(obs_shape=obs_shape, obs_dtype=obs_dtype, act_shape=act_shape, act_dtype=act_dtype,
                                num_envs=self.num_parallel_env, length=max_length, gamma=gamma, lam=lam)

    def run_one_step(self, t):
        """ Only collect dataset. No computation """
        act, logp, val = self.agent.act_batch(tf.convert_to_tensor(self.obs, dtype=tf.float32))
        act = act.numpy()
        logp = logp.numpy()
        val = val.numpy()
        obs2, rew, dones, infos = self.env.step(act)
        self.buffer.store(self.obs, act, rew, val, logp)
        self.logger.store(VVals=val)
        self.ep_ret += rew
        self.ep_len += 1

        # There are four cases there:
        # 1. if done is False. Bootstrap (truncated due to trajectory length)
        # 2. if done is True, if TimeLimit.truncated not in info. Don't bootstrap (didn't truncate)
        # 3. if done is True, if TimeLimit.truncated in info, if it is True, Bootstrap (true truncated)
        # 4. if done is True, if TimeLimit.truncated in info, if it is False. Don't bootstrap (same time)

        if t == self.steps_per_epoch - 1:
            time_truncated_dones = np.array([info.get('TimeLimit.truncated', False) for info in infos],
                                            dtype=np.bool_)
            # need to finish path for all the environments
            last_vals = self.agent.value_net.predict(obs2)
            last_vals = last_vals * np.logical_or(np.logical_not(dones), time_truncated_dones)
            self.buffer.finish_path(dones=np.ones(shape=self.num_parallel_env, dtype=np.bool_),
                                    last_vals=last_vals)
            self.logger.store(EpRet=self.ep_ret[dones], EpLen=self.ep_len[dones])
            self.obs = None
        elif np.any(dones):
            time_truncated_dones = np.array([info.get('TimeLimit.truncated', False) for info in infos],
                                            dtype=np.bool_)
            last_vals = self.agent.value_net.predict(obs2) * time_truncated_dones
            self.buffer.finish_path(dones=dones,
                                    last_vals=last_vals)
            self.logger.store(EpRet=self.ep_ret[dones], EpLen=self.ep_len[dones])
            self.ep_ret[dones] = 0.
            self.ep_len[dones] = 0
            self.obs = self.env.reset_done()

        else:
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
             train_pi_iters=80, train_vf_iters=80,
             lam=0.97, max_ep_len=1000, target_kl=0.05, entropy_coef=1e-3, logger_path: str = None):
        # Instantiate environment
        assert batch_size % num_parallel_envs == 0

        steps_per_epoch = batch_size // num_parallel_envs

        config = locals()
        runner = Runner(seed=seed, steps_per_epoch=steps_per_epoch,
                        epochs=epochs, exp_name=None, logger_path=logger_path)
        runner.setup_env(env_name=env_name, env_fn=env_fn, num_parallel_env=num_parallel_envs,
                         asynchronous=False, num_test_episodes=None)
        runner.setup_logger(config)
        runner.setup_agent(mlp_hidden=mlp_hidden,
                           pi_lr=pi_lr,
                           vf_lr=vf_lr,
                           clip_ratio=clip_ratio,
                           entropy_coef=entropy_coef,
                           target_kl=target_kl,
                           train_pi_iters=train_pi_iters,
                           train_vf_iters=train_vf_iters)
        runner.setup_replay_buffer(max_length=steps_per_epoch, gamma=gamma, lam=lam)
        runner.run()


if __name__ == '__main__':
    run_func_as_main(Runner.main)
