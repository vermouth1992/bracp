"""
Common in the runner:
1. Setup environment
2. Setup logger
3. Setup agent
4. Run
"""

import random
import time
from abc import abstractmethod, ABC

import gym
import gym.spaces
import numpy as np
import rlutils.gym
from rlutils.logx import EpochLogger, setup_logger_kwargs
from rlutils.replay_buffers import PyUniformParallelEnvReplayBuffer
from rlutils.samplers import StepSampler
from tqdm.auto import trange, tqdm


class BaseRunner(ABC):
    def __init__(self, seed, steps_per_epoch, epochs, exp_name=None, logger_path='data'):
        self.exp_name = exp_name
        self.logger_path = logger_path
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.seed = seed
        self.global_step = 0
        self.max_seed = 10000
        self.setup_seed(seed)

    def setup_logger(self, config, tensorboard=False):
        assert self.exp_name is not None, 'Call setup_env before setup_logger if exp passed by the contructor is None.'
        logger_kwargs = setup_logger_kwargs(exp_name=self.exp_name, data_dir=self.logger_path, seed=self.seed)
        self.logger = EpochLogger(**logger_kwargs, tensorboard=tensorboard)
        self.logger.save_config(config)

    def get_action_batch_test(self, obs):
        raise NotImplementedError

    def get_action_batch_explore(self, obs):
        raise NotImplementedError

    def test_agent(self, get_action, name):
        o, d, ep_ret, ep_len = self.test_env.reset(), np.zeros(shape=self.num_test_episodes, dtype=np.bool), \
                               np.zeros(shape=self.num_test_episodes), \
                               np.zeros(shape=self.num_test_episodes, dtype=np.int64)
        t = tqdm(total=self.num_test_episodes, desc=f'Testing {name}')
        while not np.all(d):
            a = get_action(o)
            o, r, d_, _ = self.test_env.step(a)
            ep_ret = r * (1 - d) + ep_ret
            ep_len = np.ones(shape=self.num_test_episodes, dtype=np.int64) * (1 - d) + ep_len
            prev_d = d.copy()
            d = np.logical_or(d, d_)
            newly_finished = np.sum(d) - np.sum(prev_d)
            if newly_finished > 0:
                t.update(newly_finished)
        t.close()
        return dict(
            TestEpRet=ep_ret,
            TestEpLen=ep_len
        )

    def setup_seed(self, seed):
        # we set numpy seed first and use it to generate other seeds
        np.random.seed(seed)
        random.seed(self.generate_seed())

    def generate_seed(self):
        return np.random.randint(self.max_seed)

    def setup_extra(self, **kwargs):
        pass

    def setup_replay_buffer(self, **kwargs):
        pass

    @abstractmethod
    def setup_agent(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def run_one_step(self, t):
        raise NotImplementedError

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_train_begin(self):
        self.start_time = time.time()

    def on_train_end(self):
        pass

    def setup_env(self,
                  env_name,
                  env_fn=None,
                  act_lim=1.,
                  num_parallel_env=1,
                  asynchronous=False,
                  num_test_episodes=None):
        """
        Environment is either created by env_name or env_fn. In addition, we apply Rescale action wrappers to
        run Pendulum-v0 using env_name from commandline.
        Other complicated wrappers should be included in env_fn.
        """

        if self.exp_name is None:
            self.exp_name = f'{env_name}_{self.__class__.__name__}_test'
        self.num_parallel_env = num_parallel_env
        self.num_test_episodes = num_test_episodes
        self.asynchronous = asynchronous
        self.env_name = env_name
        if env_fn is not None:
            self.original_env_fn = env_fn
        else:
            self.original_env_fn = lambda: gym.make(self.env_name)

        self.dummy_env = self.original_env_fn()

        wrappers = []
        # convert to 32-bit observation and action space
        if isinstance(self.dummy_env.observation_space, gym.spaces.Box):
            if self.dummy_env.observation_space.dtype == np.float64:
                print('Truncating observation_space dtype from np.float64 to np.float32')
                fn = lambda env: gym.wrappers.TransformObservation(env, f=lambda x: x.astype(np.float32))
                wrappers.append(fn)
        elif isinstance(self.dummy_env.observation_space, gym.spaces.Discrete):
            if self.dummy_env.observation_space.dtype == np.int64:
                print('Truncating observation_space dtype from np.int64 to np.int32')
                fn = lambda env: gym.wrappers.TransformObservation(env, f=lambda x: x.astype(np.int32))
                wrappers.append(fn)
        else:
            raise NotImplementedError

        # if isinstance(self.dummy_env.action_space, gym.spaces.Box):
        #     assert self.dummy_env.action_space.dtype == np.float32
        # elif isinstance(self.dummy_env.action_space, gym.spaces.Discrete):
        #     assert self.dummy_env.action_space.dtype == np.int32

        if isinstance(self.dummy_env.action_space, gym.spaces.Box):
            high_all = np.all(self.dummy_env.action_space.high == act_lim)
            low_all = np.all(self.dummy_env.action_space.low == -act_lim)
            print(f'Original high: {self.dummy_env.action_space.high}, low: {self.dummy_env.action_space.low}')
            if not (high_all and low_all):
                print(f'Rescale action space to [-{act_lim}, {act_lim}]')
                fn = lambda env: gym.wrappers.RescaleAction(env, a=-act_lim, b=act_lim)
                wrappers.append(fn)

        def _make_env():
            env = self.original_env_fn()
            for wrapper in wrappers:
                env = wrapper(env)
            return env

        self.env_fn = _make_env

        VecEnv = rlutils.gym.vector.AsyncVectorEnv if asynchronous else rlutils.gym.vector.SyncVectorEnv

        self.env = VecEnv([self.env_fn for _ in range(self.num_parallel_env)])

        self.is_discrete_env = isinstance(self.env.single_action_space, gym.spaces.Discrete)
        self.env.seed(self.generate_seed())
        self.env.action_space.seed(self.generate_seed())
        if self.num_test_episodes is not None:
            self.test_env = VecEnv([self.env_fn for _ in range(self.num_test_episodes)])
            self.test_env.seed(self.generate_seed())
            self.test_env.action_space.seed(self.generate_seed())
        else:
            self.test_env = None

    def run(self):
        self.on_train_begin()
        for i in range(1, self.epochs + 1):
            self.on_epoch_begin(i)
            for t in trange(self.steps_per_epoch, desc=f'Epoch {i}/{self.epochs}'):
                self.run_one_step(t)
                self.global_step += 1
            self.on_epoch_end(i)
        self.on_train_end()

    @classmethod
    def main(cls, *args, **kwargs):
        raise NotImplementedError

    @property
    def obs_data_spec(self):
        return self.env.single_observation_space

    @property
    def act_data_spec(self):
        return self.env.single_action_space

    def save_checkpoint(self, path=None):
        pass

    def load_checkpoint(self, path=None):
        pass


class OffPolicyRunner(BaseRunner):
    def setup_replay_buffer(self,
                            replay_size,
                            batch_size):

        data_spec = {
            'obs': self.obs_data_spec,
            'act': self.act_data_spec,
            'next_obs': self.obs_data_spec,
            'rew': gym.spaces.Space(shape=None, dtype=np.float32),
            'done': gym.spaces.Space(shape=None, dtype=np.float32)
        }
        self.replay_buffer = PyUniformParallelEnvReplayBuffer(data_spec=data_spec,
                                                              capacity=replay_size,
                                                              batch_size=batch_size,
                                                              num_parallel_env=self.num_parallel_env)

    def setup_agent(self, agent_cls, **kwargs):
        self.agent = agent_cls(obs_spec=self.obs_data_spec, act_spec=self.act_data_spec, **kwargs)
        self.agent.set_logger(self.logger)

    def setup_extra(self,
                    start_steps,
                    update_after,
                    update_every,
                    update_per_step,
                    policy_delay):
        self.update_after = update_after
        self.update_per_step = update_per_step
        self.policy_delay = policy_delay
        self.update_every = update_every
        self.sampler = StepSampler(env=self.env, start_steps=start_steps, num_steps=update_every,
                                   collect_fn=self.get_action_batch_explore)
        self.sampler.set_logger(logger=self.logger)

    def run_one_step(self, t):
        transitions = self.sampler.sample()
        for data in transitions:
            self.replay_buffer.add(data=data)
        # Update handling
        if self.sampler.total_env_steps >= self.update_after:
            for j in range(self.update_every * self.update_per_step):
                batch = self.replay_buffer.sample()
                batch['update_target'] = self.update_target == self.policy_delay - 1
                self.agent.train_on_batch(data=batch)
                self.update_target = (self.update_target + 1) % self.policy_delay

    def on_epoch_end(self, epoch):
        info = self.test_agent(get_action=self.get_action_batch_test, name=self.agent.__class__.__name__)
        self.logger.store(**info)

        # Log info about epoch
        self.logger.log_tabular('Epoch', epoch)
        self.sampler.log_tabular()
        self.logger.log_tabular('TestEpRet', with_min_and_max=True)
        self.logger.log_tabular('TestEpLen', average_only=True)
        self.agent.log_tabular()
        self.logger.log_tabular('Time', time.time() - self.start_time)
        self.logger.dump_tabular()

    def on_train_begin(self):
        self.o = self.env.reset()
        self.ep_ret = np.zeros(shape=self.num_parallel_env)
        self.ep_len = np.zeros(shape=self.num_parallel_env, dtype=np.int64)
        self.update_target = 0
        super(OffPolicyRunner, self).on_train_begin()

    @classmethod
    def main(cls,
             env_name,
             env_fn=None,
             steps_per_epoch=5000,
             epochs=200,
             start_steps=10000,
             update_after=4000,
             update_every=1,
             update_per_step=1,
             policy_delay=1,
             batch_size=256,
             num_parallel_env=1,
             num_test_episodes=30,
             seed=1,
             # agent args
             agent_cls=None,
             agent_kwargs={},
             # replay
             replay_size=int(1e6),
             logger_path=None
             ):
        config = locals()

        runner = cls(seed=seed, steps_per_epoch=steps_per_epoch // num_parallel_env, epochs=epochs,
                     exp_name=None, logger_path=logger_path)
        runner.setup_env(env_name=env_name, env_fn=env_fn, act_lim=1.0, num_parallel_env=num_parallel_env,
                         asynchronous=False, num_test_episodes=num_test_episodes)
        runner.setup_logger(config=config)

        runner.setup_agent(agent_cls=agent_cls, **agent_kwargs)
        runner.setup_extra(start_steps=start_steps,
                           update_after=update_after,
                           update_every=update_every,
                           update_per_step=update_per_step,
                           policy_delay=policy_delay)
        runner.setup_replay_buffer(replay_size=replay_size,
                                   batch_size=batch_size)

        runner.run()


class OfflineRunner(BaseRunner):
    def setup_replay_buffer(self,
                            batch_size,
                            dataset=None,
                            reward_scale=True):
        def rescale(x):
            return (x - np.min(x)) / (np.max(x) - np.min(x))

        if dataset is None:
            # modify d4rl keys
            import d4rl
            dataset = d4rl.qlearning_dataset(env=self.dummy_env)
            dataset['obs'] = dataset.pop('observations').astype(np.float32)
            dataset['act'] = dataset.pop('actions').astype(np.float32)
            dataset['next_obs'] = dataset.pop('next_observations').astype(np.float32)
            dataset['rew'] = dataset.pop('rewards').astype(np.float32)
            dataset['done'] = dataset.pop('terminals').astype(np.float32)

        if reward_scale:
            EpochLogger.log('Using reward scale', color='red')
            self.agent.reward_scale_factor = np.max(dataset['rew'] - np.min(dataset['rew']))
            EpochLogger.log(f'The scale factor is {self.agent.reward_scale_factor:.2f}')
            dataset['rew'] = rescale(dataset['rew'])

        replay_size = dataset['obs'].shape[0]
        EpochLogger.log(f'Dataset size: {replay_size}')
        self.replay_buffer = PyUniformParallelEnvReplayBuffer.from_data_dict(
            data=dataset,
            batch_size=batch_size
        )

    def setup_agent(self, agent_cls, **kwargs):
        self.agent = agent_cls(obs_spec=self.obs_data_spec, act_spec=self.act_data_spec, **kwargs)
        self.agent.set_logger(self.logger)

    def on_epoch_end(self, epoch):
        info = self.test_agent(get_action=self.get_action_batch_test, name=self.agent.__class__.__name__)
        self.logger.store(**info)

        # Log info about epoch
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('TestEpRet', with_min_and_max=True)
        self.logger.log_tabular('TestEpLen', average_only=True)
        self.agent.log_tabular()
        self.logger.log_tabular('GradientSteps', epoch * self.steps_per_epoch)
        self.logger.log_tabular('Time', time.time() - self.start_time)
        self.logger.dump_tabular()
