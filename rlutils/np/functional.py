from typing import Dict, List

import numpy as np
import scipy.signal
import sklearn

EPS = 1e-6


def gather_dict_key(infos: List[Dict], key, default=None, dtype=None):
    """ Gather a key from a list of dictionaries and return a numpy array. """
    if default is not None:
        output = np.array([info.get(key, default) for info in infos], dtype=dtype)
    else:
        output = np.array([info.get(key) for info in infos], dtype=dtype)
    return output


def flatten_dict(data: Dict):
    """

    Args:
        data: a dictionary of data

    Returns: list_data, key_to_idx

    """
    list_data = []
    key_to_idx = {}
    for i, (key, item) in enumerate(data.items()):
        list_data.append(item)
        key_to_idx[key] = i
    return list_data, key_to_idx


def shuffle_dict_data(data):
    output = {}
    list_data, key_to_index = flatten_dict(data)
    shuffled_data = sklearn.utils.shuffle(*list_data)
    for key in data:
        output[key] = shuffled_data[key_to_index[key]]
    return output


def inverse_softplus(x, beta=1.):
    assert x > 0, 'x must be positive'
    if x < 20:
        return np.log(np.exp(x * beta) - 1.) / beta
    else:
        return x


def flatten_leading_dims(array, n_dims):
    """ Flatten the leading n dims of a numpy array """
    if n_dims <= 1:
        return array
    newshape = [-1] + list(array.shape[n_dims:])
    return np.reshape(array, newshape=newshape)


def clip_arctanh(x):
    return np.arctanh(np.clip(x, a_min=-1. + EPS, a_max=1. - EPS))


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
