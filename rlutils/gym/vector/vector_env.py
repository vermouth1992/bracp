from gym.vector.vector_env import VectorEnv as GymVectorEnv


class VectorEnv(GymVectorEnv):
    def reset_done_wait(self):
        raise NotImplementedError

    def reset_done_async(self):
        raise NotImplementedError

    def reset_done(self):
        raise NotImplementedError
