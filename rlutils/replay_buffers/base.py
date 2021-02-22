"""
Abstract class for replay buffers
1. Uniform sampling dataset.
2. Prioritized replay
3. n-step return
4. Frame stack
5. Trajectory-based replay buffer for on-policy methods
"""

from abc import ABC, abstractmethod


class BaseReplayBuffer(ABC):
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def add(self, data, priority=1.0):
        raise NotImplementedError

    @property
    @abstractmethod
    def capacity(self):
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        raise NotImplementedError

    def load(self, data):
        pass

    def is_full(self):
        return len(self) == self.capacity

    def is_empty(self):
        return len(self) <= 0
