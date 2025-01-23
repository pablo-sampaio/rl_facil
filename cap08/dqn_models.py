import torch
import torch.nn as nn

import numpy as np


class DQNNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class MLP(nn.Module):
    def __init__(self, input_dim, list_hidden_dims, output_dim, final_activ_fn=None):
        super().__init__()
        layers = []
        last_dim = input_dim
        for dim in list_hidden_dims:
            layers.append( nn.Linear(last_dim, dim, bias=True) )
            layers.append( nn.ReLU() )
            last_dim = dim
        layers.append( nn.Linear(last_dim, output_dim, bias=True) )
        if final_activ_fn is not None:
            layers.append( final_activ_fn )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y

