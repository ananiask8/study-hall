"""
(Bello et al., 2019) Attention Augmented Convolutional Networks
https://arxiv.org/abs/1904.09925
"""

import torch
from torch import nn

from .aa_baseline import AAConv2d
from ._resnet import SimpleResidualBlock, Bottleneck


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if type(m) == nn.Parameter:
        torch.nn.init.kaiming_normal_(m)


class AASimpleResidualBlock(nn.Module):
    def __init__(self, c, heads, attn_size, map_size, halved=False, original=False):
        super(AASimpleResidualBlock, self).__init__()
        s = 2 if halved else 1
        self.f = nn.Sequential(
            AAConv2d(in_channels=c//s, out_channels=c, kernel_size=3, stride=1,
                     padding=1, heads=heads, attn_size=attn_size, map_size=map_size, bias=False, original=original),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=s, padding=1),
            nn.BatchNorm2d(c),
        )
        self.f.apply(init_weights)
        if halved:
            self.w = nn.Sequential(
                nn.Conv2d(in_channels=c//2, out_channels=c, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(c)
            )
            self.w.apply(init_weights)
        else:
            self.w = lambda x: x

    def forward(self, x):
        return torch.relu(self.f(x) + self.w(x))


class SimpleAAResNet(nn.Module):
    def __init__(self, n=9, original=False):
        super(SimpleAAResNet, self).__init__()
        # -1×3×32×32
        self.l0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.l0.apply(init_weights)
        # -1×16×32×32
        self.l1 = nn.Sequential(*[SimpleResidualBlock(c=16, halved=False) for _ in range(n)])
        self.l1.apply(init_weights)
        # -1×32×16×16
        l2 = [SimpleResidualBlock(c=32, halved=True)]
        l2 += [
            AASimpleResidualBlock(c=32, halved=False, heads=2, attn_size=(40, 40, 20), map_size=16, original=original)
            for _ in range(1, n)
        ]
        self.l2 = nn.Sequential(*l2)
        self.l2.apply(init_weights)
        # -1×64×8×8
        l3 = [AASimpleResidualBlock(c=64, halved=True, heads=3, attn_size=(60, 60, 30), map_size=16, original=original)]
        l3 += [AASimpleResidualBlock(c=64, halved=False, heads=3, attn_size=(60, 60, 30), map_size=8, original=original)
               for _ in range(1, n)]
        self.l3 = nn.Sequential(*l3)
        self.l3.apply(init_weights)
        # -1×64×1×1
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )
        self.classifier.apply(init_weights)

    def forward(self, x):
        y = self.gap(self.l3(self.l2(self.l1(self.l0(x))))).squeeze(-1).squeeze(-1)
        return self.classifier(y)



class AABottleneck(nn.Module):
    def __init__(self, c, heads, attn_size, map_size, halved=False):
        super(AABottleneck, self).__init__()
        s = 2 if halved else 1
        self.res = nn.Sequential(
            nn.Conv2d(in_channels=c//s, out_channels=c//4, kernel_size=1, stride=s, padding=0, bias=False),
            nn.BatchNorm2d(c//4),
            nn.ReLU(),
            AAConv2d(in_channels=c//4, out_channels=c//4, kernel_size=1, stride=1, padding=0,
                     heads=heads, attn_size=attn_size, map_size=map_size, bias=False, original=True),
            nn.BatchNorm2d(c//4),
            nn.ReLU(),
            nn.Conv2d(in_channels=c//4, out_channels=c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c),
        )
        self.res.apply(init_weights)
        if halved:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=c//2, out_channels=c, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(c)
            )
            self.skip.apply(init_weights)
        else:
            self.skip = lambda x: x

    def forward(self, x):
        return torch.relu(self.res(x) + self.skip(x))


class AAResNet(nn.Module):
    def __init__(self, sizes=tuple([3, 4, 6, 3])):
        super(AAResNet, self).__init__()
        # -1×3×32×32
        self.l0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.l0.apply(init_weights)
        # -1×64×32×32
        self.l1 = nn.Sequential(*[Bottleneck(c=256, halved=False) for _ in range(sizes[0])])
        self.l1.apply(init_weights)
        # -1×256×32×32
        self.l2 = nn.Sequential(*(
                [AABottleneck(c=512, halved=True, heads=6, attn_size=(120, 120, 24), map_size=16)] +
                [AABottleneck(c=512, halved=False, heads=6, attn_size=(120, 120, 24), map_size=16)
                 for _ in range(1, sizes[1])]
        ))
        self.l2.apply(init_weights)
        # -1×512×16×16
        self.l3 = nn.Sequential(*(
                [AABottleneck(c=1024, halved=True, heads=6, attn_size=(120, 120, 48), map_size=8)] +
                [AABottleneck(c=1024, halved=False, heads=6, attn_size=(120, 120, 48), map_size=8)
                 for _ in range(1, sizes[2])]
        ))
        self.l3.apply(init_weights)
        # -1×1024×8×8
        self.l4 = nn.Sequential(*(
                [AABottleneck(c=2048, halved=True, heads=6, attn_size=(120, 120, 96), map_size=4)] +
                [AABottleneck(c=2048, halved=False, heads=6, attn_size=(120, 120, 96), map_size=4)
                 for _ in range(1, sizes[3])]
        ))
        self.l4.apply(init_weights)
        # -1×2048×4×4
        self.gap = nn.AdaptiveAvgPool2d(1)
        # -1×2048×1×1
        self.classifier = nn.Sequential(
            nn.Linear(2048, 10)
        )
        self.classifier.apply(init_weights)
        # -1×10

    def forward(self, x):
        y = self.gap(self.l4(self.l3(self.l2(self.l1(self.l0(x)))))).squeeze(-1).squeeze(-1)
        return self.classifier(y)
