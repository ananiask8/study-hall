"""
(Hu et al., 2017) Squeeze-and-Excitation Networks
https://arxiv.org/abs/1709.01507
"""

import torch
from torch import nn

from .se_baseline import SqueezeAndExcite, SqueezeAndExciteOpts


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if type(m) == SqueezeAndExcite:
        torch.nn.init.kaiming_normal_(m.Fex.fc_down_r.weight)
        torch.nn.init.kaiming_normal_(m.Fex.fc_up_r.weight)


class SESimpleResidualBlock(nn.Module):
    def __init__(self, c, halved=False):
        super(SESimpleResidualBlock, self).__init__()
        s = 2 if halved else 1
        self.f = nn.Sequential(
            nn.Conv2d(in_channels=c//s, out_channels=c, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c),
            SqueezeAndExcite(SqueezeAndExciteOpts(C=c, r=4))
        )
        self.f.apply(init_weights)
        if halved:
            self.w = nn.Sequential(
                nn.Conv2d(in_channels=c//2, out_channels=c, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c)
            )
            self.w.apply(init_weights)
        else:
            self.w = lambda x: x

    def forward(self, x):
        return torch.relu(self.f(x) + self.w(x))


class SimpleSEResNet(nn.Module):
    def __init__(self, n=9):
        super(SimpleSEResNet, self).__init__()
        # -1×3×32×32
        self.l0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.l0.apply(init_weights)
        # -1×16×32×32
        self.l1 = nn.Sequential(*[SESimpleResidualBlock(c=16, halved=False) for _ in range(n)])
        self.l1.apply(init_weights)
        # -1×32×16×16
        self.l2 = nn.Sequential(*(
                [SESimpleResidualBlock(c=32, halved=True)] +
                [SESimpleResidualBlock(c=32, halved=False) for _ in range(1, n)]
        ))
        self.l2.apply(init_weights)
        # -1×64×8×8
        self.l3 = nn.Sequential(*(
                [SESimpleResidualBlock(c=64, halved=True)] +
                [SESimpleResidualBlock(c=64, halved=False) for _ in range(1, n)]
        ))
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


class SEBottleneck(nn.Module):
    def __init__(self, c, halved=False):
        super(SEBottleneck, self).__init__()
        s = 2 if halved else 1
        self.res = nn.Sequential(
            nn.Conv2d(in_channels=c//s, out_channels=c//4, kernel_size=1, stride=s, padding=0, bias=False),
            nn.BatchNorm2d(c//4),
            nn.ReLU(),
            nn.Conv2d(in_channels=c//4, out_channels=c//4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c//4),
            nn.ReLU(),
            nn.Conv2d(in_channels=c//4, out_channels=c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c),
            SqueezeAndExcite(SqueezeAndExciteOpts(C=c, r=4)),
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


class SEResNet(nn.Module):
    def __init__(self, sizes=tuple([3, 4, 6, 3])):
        super(SEResNet, self).__init__()
        # -1×3×32×32
        self.l0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.l0.apply(init_weights)
        # -1×64×32×32
        self.l1 = nn.Sequential(*[SEBottleneck(c=256, halved=False) for _ in range(sizes[0])])
        self.l1.apply(init_weights)
        # -1×256×32×32
        self.l2 = nn.Sequential(*(
                [SEBottleneck(c=512, halved=True)] +
                [SEBottleneck(c=512, halved=False) for _ in range(1, sizes[1])]
        ))
        self.l2.apply(init_weights)
        # -1×512×16×16
        self.l3 = nn.Sequential(*(
                [SEBottleneck(c=1024, halved=True)] +
                [SEBottleneck(c=1024, halved=False) for _ in range(1, sizes[2])]
        ))
        self.l3.apply(init_weights)
        # -1×1024×8×8
        self.l4 = nn.Sequential(*(
                [SEBottleneck(c=2048, halved=True)] +
                [SEBottleneck(c=2048, halved=False) for _ in range(1, sizes[3])]
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
