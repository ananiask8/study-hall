"""
(Hu et al., 2017) Squeeze-and-Excitation Networks
https://arxiv.org/abs/1709.01507
"""

import torch
from torch import nn
from typing import NamedTuple
from collections import OrderedDict


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if type(m) == SqueezeAndExcite:
        torch.nn.init.kaiming_normal_(m.Fex.fc_down_r.weight)
        torch.nn.init.kaiming_normal_(m.Fex.fc_up_r.weight)


class SqueezeAndExciteOpts(NamedTuple):
    C: int
    r: int


class SqueezeAndExcite(nn.Module):
    def __init__(self, opts: SqueezeAndExciteOpts):
        super(SqueezeAndExcite, self).__init__()
        self.Fsq = nn.AdaptiveAvgPool2d(1)
        self.Fex = nn.Sequential(OrderedDict([
            ('fc_down_r', nn.Linear(opts.C, opts.C // opts.r, bias=False)),
            ('relu', nn.ReLU(inplace=True)),
            ('fc_up_r', nn.Linear(opts.C // opts.r, opts.C, bias=False)),
            ('sigmoid', nn.Sigmoid())
        ]))
        self.opts = opts

    def forward(self, x):
        z = self.Fsq(x)
        z = z.squeeze(-1).squeeze(-1)
        s = self.Fex(z).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return s * x


class SEBaseline(nn.Module):
    def __init__(self, opts):
        super(SEBaseline, self).__init__()
        # 32x32x3
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.BatchNorm2d(256),
        )
        self.l1.apply(init_weights)

        # 7x7x16
        self.l2 = nn.Sequential(
            SqueezeAndExcite(SqueezeAndExciteOpts(C=256, r=4)),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
        )
        self.l2.apply(init_weights)

        # 2x2x32
        self.l3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=10, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1)
        )
        self.l3.apply(init_weights)

        # 1x1x10

    def forward(self, x):
        y = self.l3(self.l2(self.l1(x))).view(-1, 10)
        return torch.softmax(y, dim=1)
