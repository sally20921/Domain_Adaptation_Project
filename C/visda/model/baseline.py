import torch
import torch.nn as nn

import torch.nn.functional as F
import math


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hsz, n_layers):
        super(MLP, self).__init__()
        layers = []
        prev_dim = in_dim
        for i in range(n_layers):
            if i == n_layers - 1:
                layers.append(nn.Linear(prev_dim, out_dim))
            else:
                layers.extend([
                    nn.Linear(prev_dim, hsz),
                    nn.ReLU(True),
                    nn.Dropout(0.5)
                ])
                prev_dim = hsz

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class BaselineMLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        device = args.device
        self.device = device
        self.to(device)

    @classmethod
    def resolve_args(cls, args):
        return cls(args)

    def forward(self):
        return out
