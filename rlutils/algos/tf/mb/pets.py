import time

import gym.spaces
import numpy as np
import tensorflow as tf
from rlutils.replay_buffers import PyUniformParallelEnvReplayBuffer
from rlutils.runner import TFRunner, run_func_as_main
from rlutils.tf.nn.models import EnsembleDynamicsModel
from rlutils.tf.nn.planners import RandomShooter


class PETSAgent(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, mlp_hidden=128, num_ensembles=5, lr=1e-3,
                 horizon=10, num_particles=5, num_actions=1024):
        super(PETSAgent, self).__init__()
        self.dynamics_model = EnsembleDynamicsModel(obs_dim=obs_dim, act_dim=act_dim, mlp_hidden=mlp_hidden,
                                                    num_ensembles=num_ensembles, lr=lr, reward_fn=None,
                                                    terminate_fn=None)
        self.inference_model = self.dynamics_model.build_ts_model(horizon=horizon, num_particles=num_particles)
        self.planner = RandomShooter(inference_model=self.inference_model, horizon=horizon, num_actions=num_actions)

    def set_logger(self, logger):
        self.logger = logger
        self.dynamics_model.set_logger(logger=logger)

    def log_tabular(self):
        self.dynamics_model.log_tabular()

    def update_model(self, data, batch_size=64, num_epochs=60, patience=None,
                     validation_split=0.1, shuffle=True):
        self.dynamics_model.update(inputs=data, batch_size=batch_size, num_epochs=num_epochs, patience=patience,
                                   validation_split=validation_split, shuffle=shuffle)

    def act_batch(self, obs):
        return self.planner.act_batch(obs)


class Runner(TFRunner):
    def setup_replay_buffer(self,
                            replay_size):
        data_spec = {
            'obs': gym.spaces.Space(shape=self.env.single_observation_space.shape,
                                    dtype=np.float32),
            'act': gym.spaces.Space(shape=self.env.single_action_space.shape,
                                    dtype=np.float32),
            'next_obs': gym.spaces.Space(shape=self.env.single_observation_space.shape,
                                         dtype=np.float32),
            'rew': gym.spaces.Space(shape=None, dtype=np.float32),
            'done': gym.spaces.Space(shape=None, dtype=np.float32)
        }
        self.replay_buffer = PyUniformParallelEnvReplayBuffer(data_spec=data_spec,
                                                              capacity=replay_size,
                                                              batch_size=None,
                                                              num_parallel_env=self.num_parallel_env)

    def setup_agent(self, mlp_hidden=128, num_ensembles=5, lr=1e-3, horizon=10, num_particles=5, num_actions=1024):
        obs_dim = self.env.single_observation_space.shape[0]
        act_dim = self.env.single_action_space.shape[0]
        self.agent = PETSAgent(obs_dim=obs_dim, act_dim=act_dim,
                               mlp_hidden=mlp_hidden,
                               num_ensembles=num_ensembles,
                               lr=lr, horizon=horizon,
                               num_particles=num_particles,
                               num_actions=num_actions)
        self.agent.set_logger(self.logger)

    def setup_extra(self,
                    start_steps,
                    batch_size,
                    num_model_epochs,
                    patience,
                    validation_split):
        self.start_steps = start_steps
        self.batch_size = batch_size
        self.num_model_epochs = num_model_epochs
        self.patience = patience
        self.validation_split = validation_split

    def run_one_step(self, t):
        global_env_steps = self.global_step * self.num_parallel_env
        if global_env_steps >= self.start_steps:
            a = self.agent.act_batch(self.o).numpy()
            assert not np.any(np.isnan(a)), f'NAN action: {a}'
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

    def on_epoch_end(self, epoch):
        # update the model
        data = self.replay_buffer.get()
        self.agent.update_model(data=data, batch_size=self.batch_size, num_epochs=self.num_model_epochs,
                                patience=self.patience, validation_split=self.validation_split, shuffle=True)
        # Log info about epoch
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('TotalEnvInteracts', self.global_step * self.num_parallel_env)
        self.agent.log_tabular()
        self.logger.log_tabular('Time', time.time() - self.start_time)
        self.logger.dump_tabular()

    def on_train_begin(self):
        self.start_time = time.time()
        self.o = self.env.reset()
        self.ep_ret = np.zeros(shape=self.num_parallel_env)
        self.ep_len = np.zeros(shape=self.num_parallel_env, dtype=np.int64)

    @classmethod
    def main(cls,
             env_name,
             steps_per_epoch=400,
             epochs=200,
             start_steps=2000,
             num_parallel_env=2,
             seed=1,
             # sac args
             mlp_hidden=256,
             num_ensembles=3,
             learning_rate=1e-3,
             horizon=10,
             num_particles=5,
             num_actions=1024,
             batch_size=256,
             num_model_epochs=60,
             patience=10,
             validation_split=0.1,
             # replay
             replay_size=int(1e6),
             logger_path: str = None
             ):
        config = locals()
        runner = cls(seed=seed, steps_per_epoch=steps_per_epoch // num_parallel_env, epochs=epochs,
                     exp_name=None, logger_path=logger_path)
        runner.setup_env(env_name=env_name, num_parallel_env=num_parallel_env, frame_stack=None, wrappers=None,
                         asynchronous=False, num_test_episodes=None)
        runner.setup_logger(config=config)
        runner.setup_agent(mlp_hidden=mlp_hidden, num_ensembles=num_ensembles, lr=learning_rate,
                           horizon=horizon, num_particles=num_particles, num_actions=num_actions)
        runner.setup_extra(start_steps=start_steps,
                           batch_size=batch_size,
                           num_model_epochs=num_model_epochs,
                           patience=patience,
                           validation_split=validation_split)
        runner.setup_replay_buffer(replay_size=replay_size)

        runner.run()


if __name__ == '__main__':
    run_func_as_main(Runner.main)
