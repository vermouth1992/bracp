from abc import ABC, abstractmethod

import numpy as np
import rlutils.np as rln
from rlutils.gym.vector import VectorEnv


class Sampler(ABC):
    @abstractmethod
    def sample(self):
        pass

    @property
    @abstractmethod
    def total_env_steps(self):
        pass


class TrajectorySampler(Sampler):
    def sample(self):
        pass


class StepSampler(Sampler):
    def __init__(self, env: VectorEnv, start_steps, num_steps, collect_fn):
        self.env = env
        self.start_steps = start_steps
        self.num_steps = num_steps
        self.collect_fn = collect_fn

    @property
    def total_env_steps(self):
        return self._global_env_step

    def reset(self):
        self._global_env_step = 0
        self.o = self.env.reset()
        self.ep_ret = np.zeros(shape=self.env.num_envs)
        self.ep_len = np.zeros(shape=self.env.num_envs, dtype=np.int64)

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('TotalEnvInteracts', self._global_env_step)

    def sample(self):
        data_dict = []
        for _ in range(self.num_steps):
            if self._global_env_step >= self.start_steps:
                a = self.collect_fn(self.o)
                assert not np.any(np.isnan(a)), f'NAN action: {a}'
            else:
                a = self.env.action_space.sample()

            # Step the env
            o2, r, d, infos = self.env.step(a)
            self.ep_ret += r
            self.ep_len += 1

            timeouts = rln.gather_dict_key(infos=infos, key='TimeLimit.truncated', default=False, dtype=np.bool)
            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            true_d = np.logical_and(d, np.logical_not(timeouts))

            # Store experience to replay buffer
            data_dict.append({
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

            self._global_env_step += self.env.num_envs

        return data_dict
