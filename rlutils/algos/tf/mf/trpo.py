"""
Trust Region Policy Optimization
"""

import time
from typing import Callable

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from rlutils.replay_buffers import GAEBuffer
from rlutils.runner import TFRunner, run_func_as_main
from rlutils.tf.functional import to_numpy_or_python_type, flat_grads, get_flat_trainable_variables, \
    set_flat_trainable_variables
from rlutils.tf.nn import NormalActor, CategoricalActor
from rlutils.tf.nn.functional import build_mlp


class TRPOAgent(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, act_dtype, mlp_hidden=64,
                 delta=0.01, vf_lr=1e-3, damping_coeff=0.1, cg_iters=10, backtrack_iters=10,
                 backtrack_coeff=0.8, train_vf_iters=80, algo='npg'
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
        super(TRPOAgent, self).__init__()
        if act_dtype == np.int32:
            self.policy_net = CategoricalActor(obs_dim=obs_dim, act_dim=act_dim, mlp_hidden=mlp_hidden)
        else:
            self.policy_net = NormalActor(obs_dim=obs_dim, act_dim=act_dim, mlp_hidden=mlp_hidden)
        self.v_optimizer = tf.keras.optimizers.Adam(learning_rate=vf_lr)
        self.value_net = build_mlp(input_dim=obs_dim, output_dim=1, squeeze=True, mlp_hidden=mlp_hidden)
        self.value_net.compile(optimizer=self.v_optimizer, loss='mse')

        self.delta = delta
        self.damping_coeff = damping_coeff
        self.cg_iters = cg_iters
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.train_vf_iters = train_vf_iters
        self.algo = algo

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.logger = None

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossV', average_only=True)
        self.logger.log_tabular('KL', average_only=True)
        self.logger.log_tabular('DeltaLossPi', average_only=True)
        self.logger.log_tabular('DeltaLossV', average_only=True)
        self.logger.log_tabular('BacktrackIters', average_only=True)

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

    def _compute_kl(self, obs, old_pi):
        pi = self.get_pi_distribution(obs)
        kl_loss = tfp.distributions.kl_divergence(pi, old_pi)
        kl_loss = tf.reduce_mean(kl_loss)
        return kl_loss

    def _compute_loss_pi(self, obs, act, logp, adv):
        distribution = self.get_pi_distribution(obs)
        log_prob = distribution.log_prob(act)
        negative_approx_kl = log_prob - logp
        ratio = tf.exp(negative_approx_kl)
        surr1 = ratio * adv
        policy_loss = -tf.reduce_mean(surr1, axis=0)
        return policy_loss

    def _compute_gradient(self, obs, act, logp, adv):
        # compute pi gradients
        with tf.GradientTape() as tape:
            policy_loss = self._compute_loss_pi(obs, act, logp, adv)
        grads = tape.gradient(policy_loss, self.policy_net.trainable_variables)
        grads = flat_grads(grads)
        # flat grads
        return grads, policy_loss

    def _hessian_vector_product(self, obs, p):
        # compute Hx
        old_pi = self.get_pi_distribution(obs)
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                kl = self._compute_kl(obs, old_pi)
            inner_grads = t1.gradient(kl, self.policy_net.trainable_variables)
            # flat gradients
            inner_grads = flat_grads(inner_grads)
            kl_v = tf.reduce_sum(inner_grads * p)
        grads = t2.gradient(kl_v, self.policy_net.trainable_variables)
        grads = flat_grads(grads)
        _Avp = grads + p * self.damping_coeff
        return _Avp

    @tf.function
    def _conjugate_gradients(self, obs, b, nsteps, residual_tol=1e-10):
        """
        Args:
            Avp: a callable computes matrix vector produce. Note that vector here has NO dummy dimension
            b: A^{-1}b
            nsteps: max number of steps
            residual_tol:
        Returns:
        """
        print(f'Tracing _conjugate_gradients b={b}, nsteps={nsteps}')
        x = tf.zeros_like(b)
        r = tf.identity(b)
        p = tf.identity(b)
        rdotr = tf.tensordot(r, r, axes=1)
        for _ in tf.range(nsteps):
            _Avp = self._hessian_vector_product(obs, p)
            # compute conjugate gradient
            alpha = rdotr / tf.tensordot(p, _Avp, axes=1)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = tf.tensordot(r, r, axes=1)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    def _compute_natural_gradient(self, obs, act, logp, adv):
        print(f'Tracing _compute_natural_gradient with obs={obs}, act={act}, logp={logp}, adv={adv}')
        grads, policy_loss = self._compute_gradient(obs, act, logp, adv)
        x = self._conjugate_gradients(obs, grads, self.cg_iters)
        alpha = tf.sqrt(2. * self.delta / (tf.tensordot(x, self._hessian_vector_product(obs, x),
                                                        axes=1) + 1e-8))
        return alpha * x, policy_loss

    def _set_and_eval(self, obs, act, logp, adv, old_params, old_pi, natural_gradient, step):
        new_params = old_params - natural_gradient * step
        set_flat_trainable_variables(self.policy_net, new_params)
        loss_pi = self._compute_loss_pi(obs, act, logp, adv)
        kl_loss = self._compute_kl(obs, old_pi)
        return kl_loss, loss_pi

    @tf.function
    def _update_actor(self, obs, act, adv):
        print(f'Tracing _update_actor with obs={obs}, act={act}, adv={adv}')
        old_params = get_flat_trainable_variables(self.policy_net)
        old_pi = self.get_pi_distribution(obs)
        logp = old_pi.log_prob(act)
        natural_gradient, pi_l_old = self._compute_natural_gradient(obs, act, logp, adv)

        if self.algo == 'npg':
            # npg has no backtracking or hard kl constraint enforcement
            kl, pi_l_new = self._set_and_eval(obs, act, logp, adv, old_params, old_pi,
                                              natural_gradient, step=1.)
            j = tf.constant(value=0, dtype=tf.int32)
        elif self.algo == 'trpo':
            # trpo augments npg with backtracking line search, hard kl
            pi_l_new = tf.zeros(shape=(), dtype=tf.float32)
            kl = tf.zeros(shape=(), dtype=tf.float32)
            for j in tf.range(self.backtrack_iters):
                steps = tf.pow(self.backtrack_coeff, tf.cast(j, dtype=tf.float32))
                kl, pi_l_new = self._set_and_eval(obs, act, logp, adv, old_params, old_pi,
                                                  natural_gradient, step=steps)
                if kl <= self.delta and pi_l_new <= pi_l_old:
                    tf.print('Accepting new params at step', j, 'of line search.')
                    break

                if j == self.backtrack_iters - 1:
                    tf.print('Line search failed! Keeping old params.')
                    kl, pi_l_new = self._set_and_eval(obs, act, logp, adv, old_params, old_pi,
                                                      natural_gradient, step=0.)
        info = dict(
            LossPi=pi_l_old, KL=kl,
            DeltaLossPi=(pi_l_new - pi_l_old),
            BacktrackIters=j
        )
        return info

    def update_policy(self, obs, act, ret, adv, logp):
        assert tf.is_tensor(obs), f'obs must be a tf tensor. Got {obs}'
        info = self._update_actor(obs, act, adv)

        # train the value network
        v_l_old = self.value_net.evaluate(x=obs, y=ret, verbose=False)
        for i in range(self.train_vf_iters):
            loss_v = self.value_net.train_on_batch(x=obs, y=ret)

        info['LossV'] = v_l_old
        info['DeltaLossV'] = loss_v - v_l_old

        # Log changes from update
        self.logger.store(**to_numpy_or_python_type(info))


class Runner(TFRunner):
    def setup_agent(self,
                    mlp_hidden,
                    delta,
                    vf_lr,
                    damping_coeff,
                    cg_iters,
                    backtrack_iters,
                    backtrack_coeff,
                    train_vf_iters,
                    algo):
        # Instantiate policy
        obs_dim = self.env.single_observation_space.shape[0]
        act_dim = self.env.single_action_space.n if self.is_discrete_env else self.env.single_action_space.shape[0]
        act_dtype = np.int32 if self.is_discrete_env else np.float32
        self.agent = TRPOAgent(obs_dim=obs_dim, act_dim=act_dim, act_dtype=act_dtype, mlp_hidden=mlp_hidden,
                               delta=delta, vf_lr=1e-3, damping_coeff=damping_coeff, cg_iters=cg_iters,
                               backtrack_iters=backtrack_iters,
                               backtrack_coeff=backtrack_coeff, train_vf_iters=train_vf_iters, algo=algo,
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
    def main(cls, env_name, env_fn: Callable = None, mlp_hidden=128, seed=0, num_parallel_envs=5,
             batch_size=5000, epochs=200, gamma=0.99, delta=0.01, vf_lr=1e-3,
             train_vf_iters=80, damping_coeff=0.1, cg_iters=10, backtrack_iters=10,
             backtrack_coeff=0.8, lam=0.97, algo='trpo', logger_path: str = None):
        # Instantiate environment
        assert batch_size % num_parallel_envs == 0

        steps_per_epoch = batch_size // num_parallel_envs

        config = locals()
        runner = cls(seed=seed, steps_per_epoch=steps_per_epoch,
                     epochs=epochs, exp_name=None, logger_path=logger_path)
        runner.setup_env(env_name=env_name, env_fn=env_fn, num_parallel_env=num_parallel_envs,
                         asynchronous=False, num_test_episodes=None)
        runner.setup_logger(config)
        runner.setup_agent(mlp_hidden,
                           delta,
                           vf_lr,
                           damping_coeff,
                           cg_iters,
                           backtrack_iters,
                           backtrack_coeff,
                           train_vf_iters,
                           algo)
        runner.setup_replay_buffer(max_length=steps_per_epoch, gamma=gamma, lam=lam)
        runner.run()


if __name__ == '__main__':
    run_func_as_main(Runner.main)
