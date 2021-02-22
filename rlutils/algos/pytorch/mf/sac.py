"""
Implement soft actor critic agent here
"""

import copy
import time

import numpy as np
import torch
from rlutils.pytorch.functional import soft_update
from rlutils.pytorch.nn import LagrangeLayer, SquashedGaussianMLPActor, EnsembleMinQNet
from rlutils.replay_buffers import PyUniformParallelEnvReplayBuffer
from rlutils.runner import PytorchRunner, run_func_as_main
from torch import nn
from tqdm.auto import tqdm


class SACAgent(nn.Module):
    def __init__(self,
                 ob_dim,
                 ac_dim,
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
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.policy_net = SquashedGaussianMLPActor(ob_dim, ac_dim, policy_mlp_hidden)
        self.q_network = EnsembleMinQNet(ob_dim, ac_dim, q_mlp_hidden)
        self.target_q_network = copy.deepcopy(self.q_network)
        self.alpha_net = LagrangeLayer(initial_value=alpha)

        self.policy_optimizer = torch.optim.Adam(params=self.policy_net.parameters(), lr=policy_lr)
        self.q_optimizer = torch.optim.Adam(params=self.q_network.parameters(), lr=q_lr)
        self.alpha_optimizer = torch.optim.Adam(params=self.alpha_net.parameters(), lr=alpha_lr)
        self.target_entropy = -ac_dim / 2 if target_entropy is None else target_entropy

        self.tau = tau
        self.gamma = gamma

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        self.logger.log_tabular('Q1Vals', with_min_and_max=False)
        self.logger.log_tabular('Q2Vals', with_min_and_max=False)
        self.logger.log_tabular('LogPi', average_only=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('Alpha', average_only=True)
        self.logger.log_tabular('LossAlpha', average_only=True)

    def update_target(self):
        soft_update(self.target_q_network, self.q_network, self.tau)

    def _update_nets(self, obs, actions, next_obs, done, reward):
        """ Sample a mini-batch from replay buffer and update the network

        Args:
            obs: (batch_size, ob_dim)
            actions: (batch_size, action_dim)
            next_obs: (batch_size, ob_dim)
            done: (batch_size,)
            reward: (batch_size,)

        Returns: None

        """
        with torch.no_grad():
            alpha = self.alpha_net()
            next_action, next_action_log_prob, _, _ = self.policy_net((next_obs, False))
            target_q_values = self.target_q_network((next_obs, next_action),
                                                    training=False) - alpha * next_action_log_prob
            q_target = reward + self.gamma * (1.0 - done) * target_q_values

        # q loss
        q_values = self.q_network((obs, actions), training=True)  # (num_ensembles, None)
        q_values_loss = 0.5 * torch.square(torch.unsqueeze(q_target, dim=0) - q_values)
        # (num_ensembles, None)
        q_values_loss = torch.sum(q_values_loss, dim=0)  # (None,)
        # apply importance weights
        q_values_loss = torch.mean(q_values_loss)
        self.q_optimizer.zero_grad()
        q_values_loss.backward()
        self.q_optimizer.step()

        # policy loss
        action, log_prob, _, _ = self.policy_net((obs, False))
        q_values_pi_min = self.q_network((obs, action), training=False)
        policy_loss = torch.mean(log_prob * alpha - q_values_pi_min)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        alpha = self.alpha_net()
        alpha_loss = -torch.mean(alpha * (log_prob.detach() + self.target_entropy))
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        info = dict(
            Q1Vals=q_values[0],
            Q2Vals=q_values[1],
            LogPi=log_prob,
            Alpha=alpha,
            LossQ=q_values_loss,
            LossAlpha=alpha_loss,
            LossPi=policy_loss,
        )
        return info

    def update(self, obs, act, next_obs, done, rew, update_target=True):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        rew = torch.as_tensor(rew, dtype=torch.float32, device=self.device)

        info = self._update_nets(obs, act, next_obs, done, rew)
        for key, item in info.items():
            info[key] = item.detach().cpu().numpy()
        self.logger.store(**info)

        if update_target:
            self.update_target()

    def act_batch(self, obs, deterministic):
        with torch.no_grad():
            pi_final = self.policy_net.select_action((obs, deterministic))
            return pi_final


class SACRunner(PytorchRunner):
    def get_action_batch(self, o, deterministic=False):
        return self.agent.act_batch(torch.as_tensor(o, dtype=torch.float32, device=self.agent.device),
                                    deterministic).cpu().numpy()

    def test_agent(self):
        o, d, ep_ret, ep_len = self.test_env.reset(), np.zeros(shape=self.num_test_episodes, dtype=np.bool), \
                               np.zeros(shape=self.num_test_episodes), \
                               np.zeros(shape=self.num_test_episodes, dtype=np.int64)
        t = tqdm(total=1, desc='Testing')
        while not np.all(d):
            a = self.get_action_batch(o, deterministic=True)
            o, r, d_, _ = self.test_env.step(a)
            ep_ret = r * (1 - d) + ep_ret
            ep_len = np.ones(shape=self.num_test_episodes, dtype=np.int64) * (1 - d) + ep_len
            d = np.logical_or(d, d_)
        t.update(1)
        t.close()
        self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def setup_replay_buffer(self,
                            replay_size,
                            batch_size):
        obs_dim = self.env.single_observation_space.shape[0]
        act_dim = self.env.single_action_space.shape[0]
        self.replay_buffer = PyUniformParallelEnvReplayBuffer(obs_dim=obs_dim,
                                                              act_dim=act_dim,
                                                              act_dtype=np.float32,
                                                              capacity=replay_size,
                                                              batch_size=batch_size,
                                                              num_parallel_env=self.num_parallel_env)

    def setup_agent(self,
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
        obs_dim = self.env.single_observation_space.shape[0]
        act_dim = self.env.single_action_space.shape[0]
        self.agent = SACAgent(ob_dim=obs_dim, ac_dim=act_dim,
                              policy_mlp_hidden=policy_mlp_hidden,
                              policy_lr=policy_lr,
                              q_mlp_hidden=q_mlp_hidden,
                              q_lr=q_lr,
                              alpha=alpha,
                              alpha_lr=alpha_lr,
                              tau=tau,
                              gamma=gamma,
                              target_entropy=target_entropy,
                              )
        self.agent.device = torch.device("cuda")
        self.agent.to(self.agent.device)
        self.agent.set_logger(self.logger)

    def setup_extra(self,
                    start_steps,
                    max_ep_len,
                    update_after,
                    update_every,
                    update_per_step):
        self.start_steps = start_steps
        self.max_ep_len = max_ep_len
        self.update_after = update_after
        self.update_every = update_every
        self.update_per_step = update_per_step

    def run_one_step(self, t):
        global_env_steps = self.global_step * self.num_parallel_env
        if global_env_steps >= self.start_steps:
            a = self.get_action_batch(self.o, deterministic=False)
        else:
            a = self.env.action_space.sample()

        # Step the env
        o2, r, d, _ = self.env.step(a)
        self.ep_ret += r
        self.ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        true_d = np.logical_and(d, self.ep_len != self.max_ep_len)

        # Store experience to replay buffer
        self.replay_buffer.add(data={
            'obs': self.o,
            'act': a,
            'rew': r,
            'next_obs': o2,
            'done': true_d
        })

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        self.o = o2

        # End of trajectory handling
        if np.any(d):
            self.logger.store(EpRet=self.ep_ret[d], EpLen=self.ep_len[d])
            self.ep_ret[d] = 0
            self.ep_len[d] = 0
            self.o = self.env.reset_done()

        # Update handling
        if global_env_steps >= self.update_after and global_env_steps % self.update_every == 0:
            for j in range(self.update_every * self.update_per_step):
                batch = self.replay_buffer.sample()
                self.agent.update(**batch, update_target=True)

    def on_epoch_end(self, epoch):
        self.test_agent()

        # Log info about epoch
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('TestEpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('TestEpLen', average_only=True)
        self.logger.log_tabular('TotalEnvInteracts', self.global_step * self.num_parallel_env)
        self.agent.log_tabular()
        self.logger.log_tabular('Time', time.time() - self.start_time)
        self.logger.dump_tabular()

    def on_train_begin(self):
        self.start_time = time.time()
        self.o = self.env.reset()
        self.ep_ret = np.zeros(shape=self.num_parallel_env)
        self.ep_len = np.zeros(shape=self.num_parallel_env, dtype=np.int64)


def sac(env_name,
        max_ep_len=1000,
        steps_per_epoch=5000,
        epochs=200,
        start_steps=10000,
        update_after=1000,
        update_every=1,
        update_per_step=1,
        batch_size=256,
        num_parallel_env=1,
        num_test_episodes=20,
        seed=1,
        # sac args
        nn_size=256,
        learning_rate=3e-4,
        alpha=0.2,
        tau=5e-3,
        gamma=0.99,
        # replay
        replay_size=int(1e6),
        ):
    config = locals()

    runner = SACRunner(seed=seed, steps_per_epoch=steps_per_epoch // num_parallel_env, epochs=epochs,
                       exp_name=f'{env_name}_sac_test', logger_path='data')
    runner.setup_env(env_name=env_name, num_parallel_env=num_parallel_env, frame_stack=None, wrappers=None,
                     asynchronous=False, num_test_episodes=num_test_episodes)
    runner.setup_seed(seed)
    runner.setup_logger(config=config)
    runner.setup_agent(policy_mlp_hidden=nn_size,
                       policy_lr=learning_rate,
                       q_mlp_hidden=nn_size,
                       q_lr=learning_rate,
                       alpha=alpha,
                       alpha_lr=1e-3,
                       tau=tau,
                       gamma=gamma,
                       target_entropy=None)
    runner.setup_extra(start_steps=start_steps,
                       max_ep_len=max_ep_len,
                       update_after=update_after,
                       update_every=update_every,
                       update_per_step=update_per_step)
    runner.setup_replay_buffer(replay_size=replay_size,
                               batch_size=batch_size)

    runner.run()


if __name__ == '__main__':
    run_func_as_main(sac)
