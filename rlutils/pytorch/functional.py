import numpy as np
import torch
import torch.nn as nn


def soft_update(target: nn.Module, source: nn.Module, tau):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target: nn.Module, source: nn.Module):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


def compute_target_value(reward, gamma, done, next_q):
    q_target = reward + gamma * (1.0 - done) * next_q
    return q_target


def to_numpy_or_python_type(tensors):
    """Converts a structure of `Tensor`s to `NumPy` arrays or Python scalar types.

    For each tensor, it calls `tensor.numpy()`. If the result is a scalar value,
    it converts it to a Python type, such as a float or int, by calling
    `result.item()`.

    Numpy scalars are converted, as Python types are often more convenient to deal
    with. This is especially useful for bfloat16 Numpy scalars, which don't
    support as many operations as other Numpy values.

    Args:
      tensors: A structure of tensors.

    Returns:
      `tensors`, but scalar tensors are converted to Python types and non-scalar
      tensors are converted to Numpy arrays.
    """

    def _to_single_numpy_or_python_type(t):
        if isinstance(t, torch.Tensor):
            x = t.detach().cpu().numpy()
            return x.item() if np.ndim(x) == 0 else x
        return t  # Don't turn ragged or sparse tensors to NumPy.

    import tensorflow as tf
    return tf.nest.map_structure(_to_single_numpy_or_python_type, tensors)
