import torch

from .base import BaseRunner


class PytorchRunner(BaseRunner):
    def setup_seed(self, seed):
        super(PytorchRunner, self).setup_seed(seed)
        torch.random.manual_seed(self.generate_seed())
        torch.cuda.manual_seed_all(self.generate_seed())
        torch.backends.cudnn.benchmark = True
