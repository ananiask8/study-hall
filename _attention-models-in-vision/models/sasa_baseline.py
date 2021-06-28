"""
(Ramachandran et al., 2019) Stand-Alone Self-Attention in Vision Models
https://arxiv.org/abs/1906.05909
"""

import torch
from torch import nn
from torch.nn import functional as F
from math import sqrt
# from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange
from opt_einsum import contract as einsum


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    if type(m) == nn.Parameter:
        torch.nn.init.normal_(m, mean=0, std=1)


class RelativeEmbeddings2d(nn.Module):
    def __init__(self, extent, embedding_size):
        super(RelativeEmbeddings2d, self).__init__()

        assert type(extent) == int, 'RelativeEmbeddings2d requires integer extent'

        self.extent = extent
        self.embedding_size = embedding_size
        self.width_mat = nn.Parameter(torch.randn((1, embedding_size // 2, 1, extent, 1)), requires_grad=True)
        self.height_mat = nn.Parameter(torch.randn((1, embedding_size // 2, extent, 1, 1)), requires_grad=True)

    def forward(self, x):
        x_h, x_w = rearrange(
            x, 'N (C K1 K2) L -> N C K1 K2 L',
            C=self.embedding_size, K1=self.extent
        ).split(self.embedding_size // 2, dim=1)
        return rearrange(
            torch.cat((x_h + self.height_mat, x_w + self.width_mat), dim=1),
            'N C K1 K2 L -> N (C K1 K2) L'
        )


class SASAConv2d(nn.Module):
    """Stand-alone Self-attention 2d"""

    # (W−F+2P)/S+1
    def __init__(self, in_channels, out_channels, kernel_size, heads=4, stride=1):
        super(SASAConv2d, self).__init__()

        assert heads > 0, 'SASAConv2d requires a positive number of heads'
        assert type(kernel_size) == int, 'SASAConv2d requires integer kernel_size'
        assert out_channels % heads == 0, 'SASAConv2d requires out_channels divisible by the number of heads'

        padding = (kernel_size - 1) // 2
        self.heads = heads
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.q_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Unfold(1, 1, 0, stride),
            Rearrange('N (M D) HW -> (N HW M) () D', M=self.heads)
        )
        self.q_conv.apply(init_weights)
        self.k_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Unfold(kernel_size, 1, padding, stride),
            RelativeEmbeddings2d(extent=kernel_size, embedding_size=out_channels),
            Rearrange('N (M D KK) HW -> (N HW M) D KK', M=self.heads, KK=self.kernel_size ** 2)
        )
        self.k_conv.apply(init_weights)
        self.v_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Unfold(kernel_size, 1, padding, stride),
            Rearrange('N (M D KK) HW -> (N HW M) KK D', M=self.heads, KK=self.kernel_size ** 2)
        )
        self.v_conv.apply(init_weights)

    def forward(self, x):
        N, C, H, W = x.size()

        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        weights = F.softmax(q.bmm(k), dim=-1)
        attn_maps = weights.bmm(v)
        return rearrange(attn_maps, '(N H W M) () D -> N (M D) H W', N=N, H=H, W=W)


class SpatiallyAwareValueFeatures(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, heads, stride, mixtures=4):
        super(SpatiallyAwareValueFeatures, self).__init__()

        assert type(kernel_size) == int, 'MixtureEmbeddings2d requires integer kernel_size'

        padding = (kernel_size - 1) // 2
        self.heads = heads
        self.mixtures = mixtures
        self.kernel_size = kernel_size
        self.embedding_size = out_channels
        self.row_embeddings = nn.Parameter(torch.randn((out_channels // heads, kernel_size)), requires_grad=True)
        self.col_embeddings = nn.Parameter(torch.randn((out_channels // heads, kernel_size)), requires_grad=True)
        self.mix_embeddings = nn.Parameter(torch.randn((out_channels // heads, mixtures)), requires_grad=True)
        self.v_conv = nn.Sequential(
            nn.Conv2d(in_channels, mixtures * out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Unfold(kernel_size, 1, padding, stride)
        )

    def forward(self, x):
        v = self.v_conv(x)
        N, CKK, HW = v.size()
        H = W = int(sqrt(HW))

        assert HW % H == 0, 'Expect H == W'

        v = v.view(N, self.mixtures, self.heads, self.embedding_size // self.heads, self.kernel_size ** 2, HW)
        # v: N, mixtures, heads, embedding_size // heads, HW, U
        row_embeddings = self.mix_embeddings.permute(1, 0).matmul(self.row_embeddings).unsqueeze(2)
        col_embeddings = self.mix_embeddings.permute(1, 0).matmul(self.col_embeddings).unsqueeze(1)
        outer_sum = row_embeddings + col_embeddings
        # outer_sum: mixtures, kernel_size, kernel_size
        weights = F.softmax(outer_sum, dim=0).view(self.mixtures, self.kernel_size ** 2)
        CKK = self.embedding_size * self.kernel_size * self.kernel_size
        return einsum('mu,nmgeuf->ngeuf', weights, v).reshape(N, CKK, HW)


class SASAStem2d(nn.Module):
    # (W−F+2P)/S+1
    def __init__(self, in_channels, out_channels, kernel_size, heads=4, stride=1, mixtures=4):
        super(SASAStem2d, self).__init__()

        assert heads > 0, 'SASAStem2d requires a positive number of heads'
        assert type(kernel_size) == int, 'SASAStem2d requires integer kernel_size'
        assert out_channels % heads == 0, 'SASAStem2d requires out_channels divisible by the number of heads'

        padding = (kernel_size - 1) // 2
        self.heads = heads
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.q_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Unfold(1, 1, 0, stride)
        )
        self.q_conv.apply(init_weights)
        self.k_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Unfold(kernel_size, 1, padding, stride)
        )
        self.k_conv.apply(init_weights)
        self.v_conv = SpatiallyAwareValueFeatures(in_channels, out_channels, kernel_size, 1, stride, mixtures)
        self.v_conv.apply(init_weights)

    def forward(self, x):
        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        # split attention heads
        N, CKK, HW = k.size()
        H = W = int(sqrt(HW))

        assert HW % H == 0, 'Expect H == W'

        q = q.view(N, self.heads, self.out_channels // self.heads, HW)
        k = k.view(N, self.heads, self.out_channels // self.heads, self.kernel_size ** 2, HW)
        v = v.view(N, self.heads, self.out_channels // self.heads, self.kernel_size ** 2, HW)

        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 4, 2, 3)
        v = v.permute(0, 1, 4, 3, 2)
        weights = F.softmax(q.unsqueeze(-2).matmul(k).squeeze(-2), dim=-1)
        attn_maps = weights.unsqueeze(-2).matmul(v).squeeze(-2).permute(0, 1, 3, 2)
        # attn_maps: N, heads, C // heads, HW

        return attn_maps.reshape(N, self.out_channels, H, W)


class SASABaseline(nn.Module):
    def __init__(self, stem=True):
        super(SASABaseline, self).__init__()
        # 3x32x32
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.BatchNorm2d(256),
        )
        if stem:
            self.l1 = nn.Sequential(
                SASAStem2d(in_channels=3, out_channels=256, kernel_size=7, heads=16, stride=2, mixtures=4),
                nn.LeakyReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(256),
            )
        self.l1.apply(init_weights)
        # 32x8x8
        self.l2 = nn.Sequential(
            SASAConv2d(in_channels=256, out_channels=512, kernel_size=7, heads=32, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
        )
        self.l2.apply(init_weights)
        # 64x4x4
        self.l3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=10, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.l3.apply(init_weights)
        # 10x1x1

    def forward(self, x):
        y = self.l3(self.l2(self.l1(x))).view(-1, 10)
        return y
