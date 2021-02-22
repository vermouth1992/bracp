import numpy as np
import tensorflow as tf


class Planner(object):
    def __init__(self, inference_model, horizon=10):
        self.inference_model = inference_model
        self.horizon = horizon

    def reset(self):
        pass

    def act_batch(self, obs):
        raise NotImplementedError


class RandomShooter(Planner):
    def __init__(self, inference_model, horizon=10, num_actions=4096):
        self.num_actions = num_actions
        super(RandomShooter, self).__init__(inference_model=inference_model,
                                            horizon=horizon)

    @tf.function
    def act_batch(self, obs):
        """

        Args:
            obs (np.ndarray): (None, obs_dim)

        Returns:

        """
        batch_size = tf.shape(obs)[0]
        obs = tf.tile(obs, (self.num_actions, 1))  # (num_actions * None, obs_dim)
        obs = tf.tile(tf.expand_dims(obs, axis=1), (1, self.inference_model.num_particles, 1))
        act_seq = self.inference_model.sample_action(shape=[self.num_actions * batch_size, self.horizon])
        _, reward_seq, _ = self.inference_model(inputs=(obs, act_seq),
                                                training=False)  # (num_actions * None, horizon, num_particles)
        reward_seq = tf.reduce_mean(reward_seq, axis=-1)
        reward = tf.reduce_sum(reward_seq, axis=-1)  # (num_actions, None)
        reward = tf.reshape(reward, shape=(self.num_actions, batch_size))
        best_index = tf.argmax(reward, axis=0, output_type=tf.int32)
        best_index = tf.stack([best_index, tf.range(batch_size)], axis=-1)
        act_seq = tf.reshape(act_seq, shape=(self.num_actions, batch_size, self.horizon, -1))
        act_seq = tf.gather_nd(act_seq, indices=best_index)
        return act_seq[:, 0]
