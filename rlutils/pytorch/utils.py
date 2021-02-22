"""
Handle global pytorch device and data types
"""

import torch

device = None


def set_device(d):
    global device
    print(f'Setting global Pytorch device to {d}')
    if d == 'cuda':
        if not torch.cuda.is_available():
            print('CUDA is not available in this machine. Setting to cpu.')
            d = 'cpu'
    device = d
