"""
Implementing Latent Action Space. https://arxiv.org/abs/2011.07213
"""

import os
import time

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from rlutils.future.optimizer import get_adam_optimizer, minimize
from rlutils.logx import EpochLogger
from rlutils.replay_buffers import PyUniformParallelEnvReplayBuffer
from rlutils.runner import TFRunner
from rlutils.tf.functional import soft_update, hard_update, to_numpy_or_python_type
from rlutils.tf.nn import EnsembleMinQNet, BehaviorPolicy
from rlutils.tf.nn.functional import build_mlp
from tqdm.auto import tqdm, trange

tfd = tfp.distributions
tfl = tfp.layers


class PLASAgent(tf.keras.Model):
    def __init__(self,
                 ob_dim,
                 ac_dim,
                 behavior_mlp_hidden=256,
                 behavior_lr=1e-3,
                 policy_lr=5e-6,
                 policy_mlp_hidden=256,
                 q_mlp_hidden=256,
                 q_lr=1e-4,
                 tau=1e-3,
                 gamma=0.99,
                 beta=1.,
                 latent_threshold=2.0
                 ):
        super(PLASAgent, self).__init__()

        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.q_mlp_hidden = q_mlp_hidden
        self.behavior_policy = BehaviorPolicy(out_dist='normal', obs_dim=self.ob_dim, act_dim=self.ac_dim,
                                              mlp_hidden=behavior_mlp_hidden, beta=beta)
        self.behavior_policy.optimizer = get_adam_optimizer(lr=behavior_lr)
        self.policy_net = build_mlp(self.ob_dim, self.behavior_policy.latent_dim,
                                    mlp_hidden=policy_mlp_hidden, num_layers=3)
        self.target_policy_net = build_mlp(self.ob_dim, self.behavior_policy.latent_dim,
                                           mlp_hidden=policy_mlp_hidden, num_layers=3)
        # reset policy net learning rate
        self.policy_net.optimizer = get_adam_optimizer(lr=policy_lr)
        hard_update(self.target_policy_net, self.policy_net)
        self.q_network = EnsembleMinQNet(ob_dim, ac_dim, q_mlp_hidden)
        self.q_network.compile(optimizer=get_adam_optimizer(q_lr))
        self.target_q_network = EnsembleMinQNet(ob_dim, ac_dim, q_mlp_hidden)
        hard_update(self.target_q_network, self.q_network)

        self.tau = tau
        self.gamma = gamma
        self.latent_threshold = latent_threshold

    def get_action(self, policy_net, obs):
        z = policy_net(obs)
        z = tf.tanh(z) * self.latent_threshold
        raw_action = self.behavior_policy.decode_sample(z=(z, obs))
        action = tf.tanh(raw_action)
        return action, z

    def call(self, inputs, training=None, mask=None):
        obs, deterministic = inputs
        pi_final = self.policy_net((obs, deterministic))[0]
        return pi_final

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        self.logger.log_tabular('Q1Vals', with_min_and_max=True)
        self.logger.log_tabular('Q2Vals', with_min_and_max=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('BehaviorLoss', average_only=True)
        self.logger.log_tabular('Z', with_min_and_max=True)

    @tf.function
    def update_target(self):
        soft_update(self.target_q_network, self.q_network, self.tau)
        soft_update(self.target_policy_net, self.policy_net, self.tau)

    @tf.function
    def update_actor(self, obs):
        # TODO: maybe we just follow behavior policy and keep a minimum entropy instead of the optimal one.
        # policy loss
        n = 10
        batch_size = tf.shape(obs)[0]
        obs = tf.tile(obs, (n, 1))
        with tf.GradientTape() as policy_tape:
            """ Compute the loss function of the policy that maximizes the Q function """
            print(f'Tracing _compute_surrogate_loss_pi with obs={obs}')

            policy_tape.watch(self.policy_net.trainable_variables)

            action, z = self.get_action(self.policy_net, obs)
            q_values_pi_min = self.q_network((obs, action), training=False)
            q_values_pi_min = tf.reshape(q_values_pi_min, (n, batch_size))
            q_values_pi_min = tf.reduce_mean(q_values_pi_min, axis=0)
            policy_loss = -tf.reduce_mean(q_values_pi_min, axis=0)

        minimize(policy_loss, policy_tape, self.policy_net)

        info = dict(
            LossPi=policy_loss,
            Z=tf.reshape(z, shape=(-1,))
        )

        return info

    def _compute_target_q(self, next_obs, reward, done):
        n = 10
        batch_size = tf.shape(next_obs)[0]
        next_obs = tf.tile(next_obs, (n, 1))
        next_action, _ = self.get_action(self.target_policy_net, next_obs)
        # maybe add noise?
        target_q_values = self.target_q_network((next_obs, next_action), training=False)
        target_q_values = tf.reshape(target_q_values, (n, batch_size))
        target_q_values = tf.reduce_max(target_q_values, axis=0)
        q_target = reward + self.gamma * (1.0 - done) * target_q_values
        return q_target

    def _update_q_nets(self, obs, actions, q_target):
        # q loss
        with tf.GradientTape() as q_tape:
            q_tape.watch(self.q_network.trainable_variables)
            q_values = self.q_network((obs, actions), training=True)  # (num_ensembles, None)
            q_values_loss = 0.5 * tf.square(tf.expand_dims(q_target, axis=0) - q_values)
            # (num_ensembles, None)
            q_values_loss = tf.reduce_sum(q_values_loss, axis=0)  # (None,)

        minimize(q_values_loss, q_tape, self.q_network)

        info = dict(
            Q1Vals=q_values[0],
            Q2Vals=q_values[1],
            LossQ=q_values_loss,
        )
        return info

    @tf.function
    def update_q_nets(self, obs, actions, next_obs, done, reward):
        """Normal SAC update"""
        q_target = self._compute_target_q(next_obs, reward, done)
        return self._update_q_nets(obs, actions, q_target)

    @tf.function
    def _update(self, obs, act, next_obs, done, rew):
        raw_act = self.behavior_policy.inverse_transform_action(act)
        behavior_loss = self.behavior_policy.train_on_batch(x=(raw_act, obs))['loss']
        info = self.update_q_nets(obs, act, next_obs, done, rew)
        info['BehaviorLoss'] = behavior_loss
        return info

    def update(self, obs, act, next_obs, done, rew, update_target=True):
        # TODO: use different batches to update q and actor to break correlation
        info = self._update(obs, act, next_obs, done, rew)

        if update_target:
            actor_info = self.update_actor(obs)
            # we only update alpha when policy is updated
            info.update(actor_info)
            self.update_target()

        self.logger.store(**to_numpy_or_python_type(info))

    @tf.function
    def act_batch(self, obs):
        n = 20
        batch_size = tf.shape(obs)[0]
        obs_tile = tf.tile(obs, (n, 1))
        action, _ = self.get_action(self.policy_net, obs_tile)
        q_values_pi_min = self.q_network((obs_tile, action), training=True)
        q_values_pi_min = tf.reduce_mean(q_values_pi_min, axis=0)
        idx = tf.argmax(tf.reshape(q_values_pi_min, shape=(n, batch_size)), axis=0,
                        output_type=tf.int32)  # (batch_size)
        idx = tf.stack([idx, tf.range(batch_size)], axis=-1)
        samples = tf.reshape(action, shape=(n, batch_size, self.ac_dim))
        pi_final = tf.gather_nd(samples, idx)
        return pi_final

    def pretrain_behavior_policy(self, epochs, steps_per_epoch, replay_buffer):
        EpochLogger.log(f'Training behavior policy')
        t = trange(epochs)
        for epoch in t:
            loss = []
            for _ in trange(steps_per_epoch, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
                # update q_b, pi_0, pi_b
                data = replay_buffer.sample()
                obs = data['obs']
                raw_act = self.behavior_policy.inverse_transform_action(data['act'])
                behavior_loss = self.behavior_policy.train_on_batch(x=(raw_act, obs))['loss']
                loss.append(behavior_loss)
            loss = tf.reduce_mean(loss).numpy()
            t.set_description(desc=f'Loss: {loss:.2f}')


class Runner(TFRunner):
    def test_agent(self, agent, name, logger=None):
        o, d, ep_ret, ep_len = self.env.reset(), np.zeros(shape=self.num_test_episodes, dtype=np.bool), \
                               np.zeros(shape=self.num_test_episodes), np.zeros(shape=self.num_test_episodes,
                                                                                dtype=np.int64)
        t = tqdm(total=1, desc=f'Testing {name}')
        while not np.all(d):
            a = agent.act_batch(tf.convert_to_tensor(o, dtype=tf.float32)).numpy()
            assert not np.any(np.isnan(a)), f'nan action: {a}'
            o, r, d_, _ = self.env.step(a)
            ep_ret = r * (1 - d) + ep_ret
            ep_len = np.ones(shape=self.num_test_episodes, dtype=np.int64) * (1 - d) + ep_len
            d = np.logical_or(d, d_)
        t.update(1)
        t.close()
        normalized_ep_ret = self.dummy_env.get_normalized_score(ep_ret) * 100

        if logger is not None:
            logger.store(TestEpRet=ep_ret, NormalizedTestEpRet=normalized_ep_ret, TestEpLen=ep_len)
        else:
            print(f'EpRet: {np.mean(ep_ret):.2f}, TestEpLen: {np.mean(ep_len):.2f}')

    def setup_replay_buffer(self,
                            batch_size,
                            reward_scale=True):
        import d4rl
        def rescale(x):
            return (x - np.min(x)) / (np.max(x) - np.min(x))

        self.dummy_env = gym.make(self.env_name)
        dataset = d4rl.qlearning_dataset(env=self.dummy_env)

        if reward_scale:
            EpochLogger.log('Using reward scale', color='red')
            self.agent.reward_scale_factor = np.max(dataset['rewards'] - np.min(dataset['rewards']))
            EpochLogger.log(f'The scale factor is {self.agent.reward_scale_factor:.2f}')
            dataset['rewards'] = rescale(dataset['rewards'])
        # modify keys
        dataset['obs'] = dataset.pop('observations').astype(np.float32)
        dataset['act'] = dataset.pop('actions').astype(np.float32)
        dataset['obs2'] = dataset.pop('next_observations').astype(np.float32)
        dataset['rew'] = dataset.pop('rewards').astype(np.float32)
        dataset['done'] = dataset.pop('terminals').astype(np.float32)
        replay_size = dataset['obs'].shape[0]
        self.logger.log(f'Dataset size: {replay_size}')
        self.replay_buffer = PyUniformParallelEnvReplayBuffer.from_data_dict(
            data=dataset,
            batch_size=batch_size
        )

    def setup_agent(self,
                    behavior_mlp_hidden,
                    behavior_lr,
                    policy_mlp_hidden,
                    q_mlp_hidden,
                    policy_lr,
                    q_lr,
                    tau,
                    gamma,
                    ):
        obs_dim = self.env.single_observation_space.shape[-1]
        act_dim = self.env.single_action_space.shape[-1]
        self.agent = PLASAgent(ob_dim=obs_dim, ac_dim=act_dim,
                               behavior_mlp_hidden=behavior_mlp_hidden,
                               behavior_lr=behavior_lr,
                               policy_mlp_hidden=policy_mlp_hidden, q_mlp_hidden=q_mlp_hidden,
                               q_lr=q_lr, tau=tau, gamma=gamma)
        self.agent.set_logger(self.logger)
        self.behavior_filepath = os.path.join(self.logger.output_dir, 'behavior.ckpt')
        self.final_filepath = os.path.join(self.logger.output_dir, 'agent_final.ckpt')

    def setup_extra(self,
                    pretrain_epochs,
                    save_freq,
                    force_pretrain_behavior,
                    generalization_threshold,
                    ):
        self.pretrain_epochs = pretrain_epochs
        self.save_freq = save_freq
        self.force_pretrain_behavior = force_pretrain_behavior
        self.generalization_threshold = generalization_threshold

    def run_one_step(self, t):
        self.agent.update(self.replay_buffer)

    def on_epoch_end(self, epoch):
        self.test_agent(agent=self.agent, name='policy', logger=self.logger)

        # Log info about epoch
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('TestEpRet', with_min_and_max=True)
        self.logger.log_tabular('NormalizedTestEpRet', average_only=True)
        self.logger.log_tabular('TestEpLen', average_only=True)
        self.agent.log_tabular()
        self.logger.log_tabular('GradientSteps', epoch * self.steps_per_epoch)
        self.logger.log_tabular('Time', time.time() - self.start_time)
        self.logger.dump_tabular()

        if self.save_freq is not None and (epoch + 1) % self.save_freq == 0:
            self.agent.save_weights(filepath=os.path.join(self.logger.output_dir, f'agent_final_{epoch + 1}.ckpt'))

    def get_decayed_lr_schedule(self, lr, interval):
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[interval, interval * 2, interval * 3, interval * 4],
            values=[lr, 0.5 * lr, 0.1 * lr, 0.05 * lr, 0.01 * lr])
        return lr_schedule

    def on_train_begin(self):
        interval = self.pretrain_epochs * self.steps_per_epoch // 5
        behavior_lr = self.agent.behavior_lr
        self.agent.behavior_policy.optimizer = get_adam_optimizer(lr=self.get_decayed_lr_schedule(lr=behavior_lr,
                                                                                                  interval=interval))
        try:
            if self.force_pretrain_behavior:
                raise tf.errors.NotFoundError(None, None, None)
            self.agent.behavior_policy.load_weights(filepath=self.behavior_filepath).assert_consumed()
            EpochLogger.log(f'Successfully load behavior policy from {self.behavior_filepath}')
        except tf.errors.NotFoundError:
            self.agent.pretrain_behavior_policy(self.pretrain_epochs, self.steps_per_epoch, self.replay_buffer)
            self.agent.behavior_policy.save_weights(filepath=self.behavior_filepath)
        except AssertionError as e:
            print(e)
            EpochLogger.log('The structure of model is altered. Add --pretrain_behavior flag.', color='red')
            raise

        self.start_time = time.time()

    def on_train_end(self):
        self.agent.save_weights(filepath=self.final_filepath)

    @classmethod
    def main(cls,
             env_name,
             steps_per_epoch=2500,
             pretrain_epochs=200,
             pretrain_behavior=False,
             epochs=400,
             batch_size=100,
             num_test_episodes=20,
             seed=1,
             # agent args
             policy_mlp_hidden=256,
             q_mlp_hidden=256,
             policy_lr=1e-6,
             q_lr=3e-4,
             tau=5e-3,
             gamma=0.99,
             # behavior policy
             behavior_mlp_hidden=256,
             behavior_lr=1e-3,
             # others
             generalization_threshold=0.1,
             reward_scale=False,
             save_freq: int = None,
             tensorboard=False,
             ):
        """Main function to run Improved Behavior Regularized Actor Critic (BRAC+)

        Args:
            env_name (str): name of the environment
            steps_per_epoch (int): number of steps per epoch
            pretrain_epochs (int): number of epochs to pretrain
            pretrain_behavior (bool): whether to pretrain the behavior policy or load from checkpoint.
                If load fails, the flag is ignored.
            pretrain_cloning (bool):whether to pretrain the initial policy or load from checkpoint.
                If load fails, the flag is ignored.
            epochs (int): number of epochs to run
            batch_size (int): batch size of the data sampled from the dataset
            num_test_episodes (int): number of test episodes to evaluate the policy after each epoch
            seed (int): random seed
            policy_mlp_hidden (int): MLP hidden size of the policy network
            q_mlp_hidden (int): MLP hidden size of the Q network
            policy_lr (float): learning rate of the policy network
            policy_behavior_lr (float): learning rate used to train the policy that minimize the distance between the policy
                and the behavior policy. This is usally larger than policy_lr.
            q_lr (float): learning rate of the q network
            alpha_lr (float): learning rate of the alpha
            alpha (int): initial Lagrange multiplier used to control the maximum distance between the \pi and \pi_b
            tau (float): polyak average coefficient of the target update
            gamma (float): discount factor
            target_entropy (float or None): target entropy of the policy
            max_kl (float or None): maximum of the distance between \pi and \pi_b
            use_gp (bool): whether use gradient penalty or not
            reg_type (str): regularization type
            sigma (float): sigma of the Laplacian kernel for MMD
            n (int): number of samples when estimate the expectation for policy evaluation and update
            gp_weight (float): initial GP weight
            entropy_reg (bool): whether use entropy regularization or not
            kl_backup (bool): whether add the KL loss to the backup value of the target Q network
            generalization_threshold (float): generalization threshold used to compute max_kl when max_kl is None
            std_scale (float): standard deviation scale when computing target_entropy when it is None.
            num_ensembles (int): number of ensembles to train the behavior policy
            behavior_mlp_hidden (int): MLP hidden size of the behavior policy
            behavior_lr (float): the learning rate of the behavior policy
            reward_scale (bool): whether to use reward scale or not. By default, it will scale to [0, 1]
            save_freq (int or None): the frequency to save the model
            tensorboard (bool): whether to turn on tensorboard logging
        """

        config = locals()

        runner = cls(seed=seed, steps_per_epoch=steps_per_epoch, epochs=epochs,
                     exp_name=None, logger_path='data')
        runner.setup_env(env_name=env_name, num_parallel_env=num_test_episodes, frame_stack=None, wrappers=None,
                         asynchronous=False, num_test_episodes=None)
        runner.setup_logger(config=config, tensorboard=tensorboard)
        runner.setup_agent(
            behavior_mlp_hidden=behavior_mlp_hidden,
            behavior_lr=behavior_lr,
            policy_mlp_hidden=policy_mlp_hidden, q_mlp_hidden=q_mlp_hidden,
            policy_lr=policy_lr, q_lr=q_lr, tau=tau, gamma=gamma,
        )
        runner.setup_extra(pretrain_epochs=pretrain_epochs,
                           save_freq=save_freq,
                           force_pretrain_behavior=pretrain_behavior,
                           generalization_threshold=generalization_threshold
                           )

        runner.setup_replay_buffer(batch_size=batch_size,
                                   reward_scale=reward_scale)

        runner.run()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--pretrain_behavior', action='store_true')
    parser.add_argument('--pretrain_cloning', action='store_true')
    parser.add_argument('--seed', type=int, default=1)

    args = vars(parser.parse_args())
    env_name = args['env_name']

    Runner.main(**args)
