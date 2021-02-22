import os

import tensorflow as tf

from .base import BaseRunner


class TFRunner(BaseRunner):
    def setup_seed(self, seed):
        super(TFRunner, self).setup_seed(seed=seed)
        tf.random.set_seed(self.generate_seed())
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
