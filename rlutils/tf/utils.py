import tensorflow as tf

ALLOW_GROWTH = False


def set_tf_allow_growth(enable=True):
    try:
        global ALLOW_GROWTH
        if enable != ALLOW_GROWTH:
            physical_devices = tf.config.list_physical_devices('GPU')
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, enable)
            ALLOW_GROWTH = enable
        print(f'Setting Tensorflow GPU memory allow_growth={enable}')
    except RuntimeError:
        print('Unable to change Tensorflow GPU memory allow_growth option')


def get_tf_func(instance, fun):
    func = getattr(instance, fun.__name__, None)
    if func is None:
        setattr(instance, fun.__name__, fun)
        func = getattr(instance, fun.__name__, None)
    return func
