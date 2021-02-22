import numpy as np
import tensorflow as tf
from tqdm.auto import trange


class EpochLoggerCallback(tf.keras.callbacks.Callback):
    """
    Log the result every epoch instead of every step
    """

    def __init__(self, keys, epochs, logger=None, decs='Training', decimal=2):
        """

        Args:
            keys: a tuple from display key to logs key. (e.g. [(TrainLoss, loss), (ValLoss, loss)])
            epochs: total number of epochs
            logger: epoch logger or None
            decs: display message
        """
        self.logger = logger
        self.epochs = epochs
        self.keys = keys
        self.decs = decs
        self.decimal = decimal
        super(EpochLoggerCallback, self).__init__()

    def on_train_begin(self, logs=None):
        self.t = trange(self.epochs, desc=self.decs)

    def on_train_end(self, logs=None):
        self.t.close()

    def on_epoch_end(self, epoch, logs=None):
        display = []
        for display_key, log_key in self.keys:
            display.append((display_key, logs[log_key]))
            assert not np.isnan(logs[log_key])
        if self.logger is not None:
            display = {key: value for key, value in display}
            self.logger.store(**display)
        else:
            display = [f'{key}: {value:.{self.decimal}f}' for key, value in display]
            message = ', '.join(display)
            self.t.set_description(message)
        self.t.update(1)
