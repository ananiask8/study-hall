"""
(Ramachandran et al., 2019) Stand-Alone Self-Attention in Vision Models
https://arxiv.org/abs/1906.05909
"""

import torch
from torch import nn
from torch.nn import functional as F
from math import sqrt
# from torch import einsum
from opt_einsum import contract as einsum

from .sasa_baseline import SASAConv2d, SASAStem2d
from ._resnet import SimpleResidualBlock, Bottleneck


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    if type(m) == nn.Parameter:
        torch.nn.init.normal_(m, mean=0, std=1)


class SASASimpleResidualBlock(nn.Module):
    def __init__(self, c, heads, halved=False):
        super(SASASimpleResidualBlock, self).__init__()
        s = 2 if halved else 1
        self.f = nn.Sequential(
            nn.Conv2d(in_channels=c//s, out_channels=c, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            SASAConv2d(in_channels=c, out_channels=c, kernel_size=7, stride=1, heads=heads),
            nn.BatchNorm2d(c),
        )
        self.f.apply(init_weights)
        if halved:
            self.w = nn.Sequential(
                nn.Conv2d(in_channels=c // 2, out_channels=c, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(c)
            )
            self.w.apply(init_weights)
        else:
            self.w = lambda x: x

    def forward(self, x):
        return torch.relu(self.f(x) + self.w(x))


class SimpleSASAResNet(nn.Module):
    def __init__(self, n=9, stem=False):
        super(SimpleSASAResNet, self).__init__()
        # -1×3×32×32
        self.l0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        if stem:
            self.l0 = nn.Sequential(
                SASAStem2d(in_channels=3, out_channels=16, kernel_size=5, heads=1, stride=1, mixtures=4),
                nn.BatchNorm2d(16),
                nn.ReLU()
            )
        self.l0.apply(init_weights)
        # -1×16×32×32
        self.l1 = nn.Sequential(*[SimpleResidualBlock(c=16, halved=False) for _ in range(n)])
        self.l1.apply(init_weights)
        # -1×32×16×16
        l2 = [SASASimpleResidualBlock(c=32, halved=True, heads=8)]
        l2 += [
            SASASimpleResidualBlock(c=32, halved=False, heads=8)
            for _ in range(1, n)
        ]
        self.l2 = nn.Sequential(*l2)
        self.l2.apply(init_weights)
        # -1×64×8×8
        l3 = [SASASimpleResidualBlock(c=64, halved=True, heads=8)]
        l3 += [SASASimpleResidualBlock(c=64, halved=False, heads=8)
               for _ in range(1, n)]
        self.l3 = nn.Sequential(*l3)
        self.l3.apply(init_weights)
        # -1×64×1×1
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 10)
        )
        self.classifier.apply(init_weights)

    def forward(self, x):
        y = self.gap(self.l3(self.l2(self.l1(self.l0(x))))).squeeze(-1).squeeze(-1)
        return self.classifier(y)
