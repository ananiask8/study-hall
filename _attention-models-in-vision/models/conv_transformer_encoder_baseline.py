import torch
from torch import nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


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


class ConvMGSA(nn.Module):
    """ConvMGSA: Convolutional Multi-Headed Group Self-Attention"""

    def __init__(self, in_channels=256, out_channels=256, kernel_size=3, groups=35, heads=4, stride=1):
        super(ConvMGSA, self).__init__()

        assert heads > 0, 'ConvMGSA requires a positive number of heads'
        assert type(kernel_size) == int, 'ConvMGSA requires integer kernel_size'
        assert out_channels % heads == 0, 'ConvMGSA requires out_channels divisible by the number of heads'

        padding = (kernel_size - 1) // 2
        self.heads = heads
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        self.input_rearrange = Rearrange('N G (M D) H W -> N (G M D) H W', M=heads)
        self.skip_rearrange = Rearrange('N G (M D) H W -> N (H W) M G D', M=heads)
        self.q_conv = nn.Sequential(
            nn.Conv2d(groups * in_channels, groups * out_channels,
                      kernel_size=1, stride=1, padding=0, bias=False, groups=groups),
            nn.Unfold(1, 1, 0, stride),
            Rearrange('N (G M D) HW -> (N HW M) G D', G=groups, M=heads)
        )
        self.k_conv = nn.Sequential(
            nn.Conv2d(groups * in_channels, groups * out_channels,
                      kernel_size=1, stride=1, padding=0, bias=False, groups=groups),
            nn.Unfold(kernel_size, 1, padding, stride),
            RelativeEmbeddings2d(extent=kernel_size, embedding_size=groups * out_channels),
            Rearrange('N (G M D KK) HW -> (N HW M) D (G KK)', G=groups, M=heads, KK=kernel_size ** 2)
        )
        self.v_conv = nn.Sequential(
            nn.Conv2d(groups * in_channels, groups * out_channels,
                      kernel_size=1, stride=1, padding=0, bias=False, groups=groups),
            nn.Unfold(kernel_size, 1, padding, stride),
            Rearrange('N (G M D KK) HW -> (N HW M) (G KK) D', G=groups, M=heads, KK=kernel_size ** 2)
        )
        self.conv = nn.Conv2d(groups * out_channels, groups * in_channels,
                              kernel_size=1, stride=1, padding=0, bias=True, groups=groups)
        self.ffnn = nn.Sequential(
            nn.Conv2d(groups * in_channels, groups * out_channels,
                      kernel_size=1, stride=1, padding=0, bias=True, groups=groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(groups * out_channels, groups * in_channels,
                      kernel_size=1, stride=1, padding=0, bias=True, groups=groups),
        )
        self.ln1 = nn.LayerNorm(in_channels // heads)
        self.ln2 = nn.LayerNorm(in_channels // heads)

    def forward(self, x):
        N, G, C, H, W = x.size()
        x_rearranged = self.input_rearrange(x)
        q = self.q_conv(x_rearranged)
        k = self.k_conv(x_rearranged)
        v = self.v_conv(x_rearranged)

        attention_weights = F.softmax(q.bmm(k), dim=-1)
        attention_values = attention_weights.bmm(v)
        y = attention_values.view(N, H*W, self.heads, G, -1).permute(0, 3, 2, 4, 1).reshape(N, G, -1, H, W)
        y = self.conv(y.view(N, -1, H, W)).view(N, G, -1, H, W)
        y = self.ln1(self.skip_rearrange(x) + self.skip_rearrange(y)).reshape(N, G, -1, H, W)
        return self.ln2(
            self.skip_rearrange(y) + self.skip_rearrange(self.ffnn(y.view(N, -1, H, W)).view(N, G, -1, H, W))
        ).reshape(N, G, -1, H, W)


class TensorsByPatches(nn.Module):
    def __init__(self, factor=4):
        super(TensorsByPatches, self).__init__()
        self.factor = factor

    def forward(self, x):
        assert x.size(-1) % self.factor == 0 and x.size(-2) % self.factor == 0, \
          f'Both height and width must be divisible by {self.factor}'
        kw = x.size(-1) // self.factor
        kh = x.size(-2) // self.factor
        c = x.size(-3)
        x_by_patches = F.unfold(x, kernel_size=(kh, kw), stride=(kh, kw))
        return rearrange(x_by_patches, 'N (C H W) T -> T N C H W', C=c, H=kh, W=kw)


class ConvTransformerEncoderBaseline(nn.Module):
    def __init__(self):
        super(ConvTransformerEncoderBaseline, self).__init__()
        # Nx3x32x32
        self.backbone = nn.Sequential(
            TensorsByPatches(factor=4),
            ConvMGSA(in_channels=3, out_channels=256, kernel_size=3, groups=4*4, heads=8, stride=1)
        )
        # Nx16x256x8x8
        self.classifier = nn.Sequential(
            Rearrange('N T C H W -> N (T C) H W'),
            nn.AdaptiveAvgPool2d(1),
            Rearrange('N TC () () -> N TC'),
            nn.Dropout(0.2),
            nn.Linear(16 * 256, 1, bias=True),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))
