import torch
import torch.nn as nn

from rlutils.pytorch.nn.functional import build_mlp


class EnsembleMinQNet(nn.Module):
    def __init__(self, ob_dim, ac_dim, mlp_hidden, num_ensembles=2, num_layers=3):
        super(EnsembleMinQNet, self).__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.mlp_hidden = mlp_hidden
        self.num_ensembles = num_ensembles
        self.q_net = build_mlp(input_dim=self.ob_dim + self.ac_dim,
                               output_dim=1,
                               mlp_hidden=self.mlp_hidden,
                               num_ensembles=self.num_ensembles,
                               num_layers=num_layers,
                               squeeze=True)

    def forward(self, inputs, training=None):
        assert training is not None
        obs, act = inputs
        inputs = torch.cat((obs, act), dim=-1)
        inputs = torch.unsqueeze(inputs, dim=0)  # (1, None, obs_dim + act_dim)
        inputs = inputs.repeat(self.num_ensembles, 1, 1)
        q = self.q_net(inputs)  # (num_ensembles, None)
        if training:
            return q
        else:
            return torch.min(q, dim=0)[0]
