import torch.nn as nn

from .layers import EnsembleDense, SqueezeLayer

str_to_activation = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'softplus': nn.Softplus,
}


def decode_activation(activation):
    if isinstance(activation, str):
        act_fn = str_to_activation.get(activation)
    elif callable(activation):
        act_fn = activation
    elif activation is None:
        act_fn = nn.Identity
    else:
        raise ValueError('activation must be a string or callable.')
    return act_fn


def build_mlp(input_dim, output_dim, mlp_hidden, num_ensembles=None, num_layers=3,
              activation='relu', out_activation=None, squeeze=False, dropout=None,
              batch_norm=False):
    assert not batch_norm, 'BatchNorm is not supported yet.'
    activation_fn = decode_activation(activation)
    output_activation_fn = decode_activation(out_activation)
    layers = []
    if num_layers == 1:
        if num_ensembles is not None:
            layers.append(EnsembleDense(num_ensembles, input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, output_dim))
    else:
        # first layer
        if num_ensembles is not None:
            layers.append(EnsembleDense(num_ensembles, input_dim, mlp_hidden))
        else:
            layers.append(nn.Linear(input_dim, mlp_hidden))
        layers.append(activation_fn())
        if dropout is not None:
            layers.append(nn.Dropout(p=dropout))

        # intermediate layers
        for _ in range(num_layers - 2):
            if num_ensembles is not None:
                layers.append(EnsembleDense(num_ensembles, mlp_hidden, mlp_hidden))
            else:
                layers.append(nn.Linear(mlp_hidden, mlp_hidden))
            layers.append(activation_fn())
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout))

        # final dense layer
        if num_ensembles is not None:
            layers.append(EnsembleDense(num_ensembles, mlp_hidden, output_dim))
        else:
            layers.append(nn.Linear(mlp_hidden, output_dim))

    if out_activation is not None:
        layers.append(output_activation_fn())
    if output_dim == 1 and squeeze is True:
        layers.append(SqueezeLayer(dim=-1))
    model = nn.Sequential(*layers)
    return model
