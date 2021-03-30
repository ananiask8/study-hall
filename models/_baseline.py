import torch
from torch import nn


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


class Baseline(nn.Module):
    def __init__(self, opts):
        super(Baseline, self).__init__()
        # 3x32x32
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.BatchNorm2d(256)
        )
        self.l1.apply(init_weights)

        # 16x7x7
        self.l2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512)
        )
        self.l2.apply(init_weights)

        # 32x2x2
        self.l3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=10, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1)
        )
        self.l3.apply(init_weights)
        # 1x1x10

    def forward(self, x):
        y = self.l3(self.l2(self.l1(x))).view(-1, 10)
        return torch.softmax(y, dim=1)
