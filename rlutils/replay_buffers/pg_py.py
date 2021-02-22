"""
Generalized advantage estimation buffer
"""

import numpy as np
from rlutils.np.functional import discount_cumsum
from rlutils.np.functional import flatten_leading_dims

from .utils import combined_shape


class GAEBuffer(object):
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_shape, obs_dtype, act_shape, act_dtype, num_envs, length, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(shape=combined_shape(num_envs, (length, *obs_shape)), dtype=obs_dtype)
        self.act_buf = np.zeros(shape=combined_shape(num_envs, (length, *act_shape)), dtype=act_dtype)
        self.adv_buf = np.zeros(shape=(num_envs, length), dtype=np.float32)
        self.rew_buf = np.zeros(shape=(num_envs, length), dtype=np.float32)
        self.ret_buf = np.zeros(shape=(num_envs, length), dtype=np.float32)
        self.val_buf = np.zeros(shape=(num_envs, length), dtype=np.float32)
        self.logp_buf = np.zeros(shape=(num_envs, length), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.num_envs = num_envs
        self.max_size = length
        self.reset()

    def reset(self):
        self.ptr, self.path_start_idx = 0, np.zeros(shape=(self.num_envs), dtype=np.int32)

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[:, self.ptr] = obs
        self.act_buf[:, self.ptr] = act
        self.rew_buf[:, self.ptr] = rew
        self.val_buf[:, self.ptr] = val
        self.logp_buf[:, self.ptr] = logp
        self.ptr += 1

    def finish_path(self, dones, last_vals):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        for i in range(self.num_envs):
            if dones[i]:
                path_slice = slice(self.path_start_idx[i], self.ptr)
                rews = np.append(self.rew_buf[i, path_slice], last_vals[i])
                vals = np.append(self.val_buf[i, path_slice], last_vals[i])

                # the next two lines implement GAE-Lambda advantage calculation
                deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
                self.adv_buf[i, path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

                # the next line computes rewards-to-go, to be targets for the value function
                self.ret_buf[i, path_slice] = discount_cumsum(rews, self.gamma)[:-1]

                self.path_start_idx[i] = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        assert np.all(self.path_start_idx == self.ptr)
        self.reset()
        # ravel the data
        obs_buf = flatten_leading_dims(self.obs_buf, n_dims=2)
        act_buf = flatten_leading_dims(self.act_buf, n_dims=2)
        ret_buf = flatten_leading_dims(self.ret_buf, n_dims=2)
        adv_buf = flatten_leading_dims(self.adv_buf, n_dims=2)
        logp_buf = flatten_leading_dims(self.logp_buf, n_dims=2)
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(adv_buf), np.std(adv_buf)
        adv_buf = (adv_buf - adv_mean) / adv_std
        data = dict(obs=obs_buf, act=act_buf, ret=ret_buf,
                    adv=adv_buf, logp=logp_buf)
        return data
