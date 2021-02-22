from collections import deque

try:
    import reverb
except:
    print('Reverb is not installed.')
import tensorflow as tf

from .base import BaseReplayBuffer


class ReverbReplayBuffer(BaseReplayBuffer):
    def __init__(self,
                 data_spec,
                 replay_capacity,
                 batch_size,
                 update_horizon=1,
                 frame_stack=1
                 ):
        """

        Args:
            data_spec: tf.TensorSpec
            replay_capacity (int): capacity of the replay buffer
            batch_size (int):
        """
        self.table_name = 'uniform_replay'
        self.table = reverb.Table(
            name=self.table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=replay_capacity,
            rate_limiter=reverb.rate_limiters.MinSize(1),
        )
        self.server = reverb.Server(
            tables=[self.table]
        )
        self.client = reverb.Client(f'localhost:{self.server.port}')

        self.frame_stack = frame_stack
        self.total_horizon = update_horizon + frame_stack

        self.replay_dataset = reverb.ReplayDataset(
            server_address=f'localhost:{self.server.port}',
            table=self.table_name,
            max_in_flight_samples_per_worker=10,
            sequence_length=self.total_horizon,
            dtypes=tf.nest.map_structure(lambda x: x.dtype, data_spec),
            shapes=tf.nest.map_structure(lambda x: x.shape, data_spec)
        )
        self.writer = self.client.writer(max_sequence_length=self.total_horizon)
        self.dataset = self.replay_dataset.batch(self.total_horizon).batch(batch_size).__iter__()
        self._num_items = 0
        self.replay_capacity = replay_capacity

    @property
    def capacity(self):
        return self.replay_capacity

    def __del__(self):
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()

    def get_table_info(self):
        return self.client.server_info()[self.table_name]

    def __len__(self):
        return self.get_table_info().current_size

    def add(self, data, priority=1.0):
        self.writer.append(data=data)
        if self._num_items >= self.total_horizon:
            self.writer.create_item(table=self.table_name, num_timesteps=self.total_horizon, priority=priority)
        else:
            self._num_items += 1

    def sample(self):
        return next(self.dataset).data


class ReverbTransitionReplayBuffer(ReverbReplayBuffer):
    def __init__(self,
                 num_parallel_env,
                 obs_spec,
                 act_spec,
                 replay_capacity,
                 batch_size,
                 gamma=0.99,
                 update_horizon=1,
                 frame_stack=1,
                 ):
        assert replay_capacity % num_parallel_env == 0, 'replay_capacity must be divisible by num_parallel_env'
        assert batch_size % num_parallel_env == 0, 'batch_size must be divisible by num_parallel_env'

        self.obs_spec = obs_spec
        self.act_spec = act_spec

        obs_spec = tf.TensorSpec(shape=[num_parallel_env] + obs_spec.shape, dtype=obs_spec.dtype)
        act_spec = tf.TensorSpec(shape=[num_parallel_env] + act_spec.shape, dtype=act_spec.dtype)

        data_spec = {
            'obs': obs_spec,
            'act': act_spec,
            'rew': tf.TensorSpec(shape=[num_parallel_env], dtype=tf.float32),
            'done': tf.TensorSpec(shape=[num_parallel_env], dtype=tf.float32)
        }
        super(ReverbTransitionReplayBuffer, self).__init__(data_spec=data_spec,
                                                           replay_capacity=replay_capacity // num_parallel_env,
                                                           batch_size=batch_size // num_parallel_env,
                                                           update_horizon=update_horizon,
                                                           frame_stack=frame_stack)
        self.gamma = gamma

        self.rew_deque = deque(maxlen=update_horizon)
        self.done_deque = deque(maxlen=update_horizon)
        for _ in range(update_horizon):
            self.rew_deque.append(tf.zeros(shape=[num_parallel_env], dtype=tf.float32))
            self.done_deque.append(tf.zeros(shape=[num_parallel_env], dtype=tf.float32))

        self.gamma_array = tf.math.cumprod(tf.ones(shape=[update_horizon, 1], dtype=tf.float32) * self.gamma,
                                           exclusive=True, axis=0)

        self.out_perm = [0, 2, 1] + list(range(3, 3 + len(self.obs_spec.shape)))

    def add(self, data, priority=1.0):
        """For n-step return, we only know the reward for state s_t in s_{t+n-1}.

        Args:
            data: a dictionary contains obs, act, rew and done
            priority:

        Returns:

        """
        rew = tf.cast(data['rew'], dtype=tf.float32)
        done = tf.cast(data['done'], dtype=tf.float32)

        self.rew_deque.append(rew)
        self.done_deque.append(done)

        rew_queue = tf.stack(list(self.rew_deque), axis=0)  # (T, B)
        not_done_queue = 1. - tf.stack(list(self.done_deque), axis=0)  # (T, B)
        not_done_cumprod = tf.math.cumprod(not_done_queue, exclusive=True, axis=0)  # (T, B)

        rew = tf.reduce_sum(rew_queue * self.gamma_array * not_done_cumprod, axis=0)
        done = 1 - tf.math.reduce_prod(not_done_queue, axis=0)

        data['rew'] = rew
        data['done'] = done

        super(ReverbTransitionReplayBuffer, self).add(data=data)

    @tf.function
    def sample(self):
        print('Tracing sample in ReverbTransitionReplayBuffer')
        data = super(ReverbTransitionReplayBuffer, self).sample()
        obs_seq = data['obs']  # (None, update_horizon + frame_stack, B, ...)
        act_seq = data['act']  # (None, update_horizon + frame_stack, B, ...)
        rew_seq = data['rew']  # (None, update_horizon + frame_stack, B, ...)
        done_seq = data['done']  # (None, update_horizon + frame_stack, B, ...)

        obs_seq = tf.transpose(obs_seq, perm=self.out_perm)  # (None, B, update_horizon + frame_stack)
        obs_seq = tf.reshape(obs_seq, shape=[-1, self.total_horizon] + list(self.obs_spec.shape))

        obs = obs_seq[:, :self.frame_stack]  # (None * B, frame_stack, ...)
        next_obs = obs_seq[:, -self.frame_stack:]  # (None * B, frame_stack, ...)
        act = act_seq[:, self.frame_stack - 1]  # (None, B)
        rew = rew_seq[:, self.total_horizon - 2]  # (None, B)
        done = done_seq[:, self.total_horizon - 2]  # (None, B)

        act = tf.reshape(act, shape=[-1] + list(self.act_spec.shape))
        rew = tf.reshape(rew, shape=[-1])
        done = tf.reshape(done, shape=[-1])

        if self.frame_stack == 1:
            obs = tf.squeeze(obs, axis=1)
            next_obs = tf.squeeze(next_obs, axis=1)

        return {
            'obs': obs,
            'act': act,
            'next_obs': next_obs,
            'rew': rew,
            'done': done
        }


if __name__ == '__main__':
    num_parallel_env = 5
    replay_buffer = ReverbTransitionReplayBuffer(num_parallel_env=num_parallel_env,
                                                 obs_spec=tf.TensorSpec(shape=[1], dtype=tf.int32),
                                                 act_spec=tf.TensorSpec(shape=[1], dtype=tf.int32),
                                                 replay_capacity=1000,
                                                 batch_size=10,
                                                 update_horizon=2,
                                                 frame_stack=1)

    for i in range(100):
        replay_buffer.add(data={
            'obs': tf.convert_to_tensor([[i]] * num_parallel_env),
            'act': tf.convert_to_tensor([[i]] * num_parallel_env),
            'rew': tf.convert_to_tensor([i] * num_parallel_env, dtype=tf.float32),
            'done': tf.convert_to_tensor([False] * num_parallel_env) if i % 4 != 0 else tf.convert_to_tensor(
                [True] * num_parallel_env)
        })

    for _ in range(10):
        replay_buffer.sample()
