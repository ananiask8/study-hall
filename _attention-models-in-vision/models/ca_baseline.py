import torch
from torch import nn
from math import sqrt


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


class SimpleChannelAttn(nn.Module):
    def __init__(self, size, mode='skip'):
        super(SimpleChannelAttn, self).__init__()
        C, H, W = size
        self.Wq = nn.Parameter(torch.zeros((H*W, H*W)), requires_grad=True)
        init_weights(self.Wq)
        self.Wk = nn.Parameter(torch.zeros((H*W, H*W)), requires_grad=True)
        init_weights(self.Wk)
        self.Wv = nn.Parameter(torch.zeros((H*W, H*W)), requires_grad=True)
        init_weights(self.Wv)
        self.f = lambda x, o: x + o if mode == 'skip' else o
        self.f = (lambda x, o: x * o) if mode == 'scale' else self.f

    def forward(self, x):
        N, C, H, W = x.size()
        encoded = x.view(N, C, H*W)
        q = encoded.bmm(self.Wq.repeat(N, 1, 1))
        k = encoded.bmm(self.Wk.repeat(N, 1, 1))
        v = encoded.bmm(self.Wv.repeat(N, 1, 1))
        o = torch.softmax(q.bmm(k.transpose(2, 1) / sqrt(C)), dim=2).bmm(v).view(N, C, H, W)
        return self.f(x, o)


class ComplexChannelAttn(nn.Module):
    def __init__(self, size, mode='none', r=1):
        super(ComplexChannelAttn, self).__init__()
        C, H, W = size
        self.Wq = nn.Sequential(
            nn.Linear(in_features=H*W, out_features=H*W//r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=H*W//r, out_features=H*W, bias=False),
            nn.Sigmoid()
        )
        self.Wq.apply(init_weights)
        self.Wk = nn.Sequential(
            nn.Linear(in_features=H*W, out_features=H*W//r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=H*W//r, out_features=H*W, bias=False),
            nn.Sigmoid()
        )
        self.Wk.apply(init_weights)
        self.Wv = nn.Sequential(
            nn.Linear(in_features=H*W, out_features=H*W//r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=H*W//r, out_features=H*W, bias=False),
            nn.Sigmoid()
        )
        self.Wv.apply(init_weights)
        self.f = lambda x, o: x + o if mode == 'skip' else o
        self.f = (lambda x, o: x * o) if mode == 'scale' else self.f

    def forward(self, x):
        N, C, H, W = x.size()
        encoded = x.view(N, C, H*W)
        q = self.Wq(encoded)
        k = self.Wk(encoded)
        v = self.Wv(encoded)
        o = torch.softmax(q.bmm(k.transpose(2, 1) / sqrt(C)), dim=2).bmm(v).view(N, C, H, W)
        return self.f(x, o)


class ChannelAttnBaseline(nn.Module):
    """
    When the ChannelAttnBaseline applies its attention block on an input with size (C, H, W)
    each channel is considered as having a feature vector of length HW, which is obtained
    by reshaping the input. Then the classic formulation of query, key and value is used;
    two options are available to obtain the projections from the input onto the common space:
     1) SimpleChannelAttn — using simple matrices Wq, Wk, Wv;
     2) ComplexChannelAttn — MLPs are used inplace of the matrices.
    """
    def __init__(self, simple=True, mode='none'):
        super(ChannelAttnBaseline, self).__init__()
        # 32x32x3
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.BatchNorm2d(256),
            SimpleChannelAttn((256, 7, 7), mode) if simple else ComplexChannelAttn((256, 7, 7), mode),
        )
        self.l1.apply(init_weights)

        # 7x7x16
        self.l2 = nn.Sequential(
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
        return y
