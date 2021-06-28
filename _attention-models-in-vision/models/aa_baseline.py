"""
(Bello et al., 2019) Attention Augmented Convolutional Networks
https://arxiv.org/abs/1904.09925
"""

import torch
from torch import nn


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
        torch.nn.init.normal_(m, mean=0, std=0.01)


class RelativeEncoding2d(nn.Module):
    def __init__(self, map_width, map_height, map_depth):
        super(RelativeEncoding2d, self).__init__()
        self.width_mat = nn.Parameter(
            (map_depth ** -0.5) * torch.randn((2 * map_width - 1, map_depth)), requires_grad=True)
        self.height_mat = nn.Parameter(
            (map_depth ** -0.5) * torch.randn((2 * map_height - 1, map_depth)), requires_grad=True)
        # init_weights(self.width_mat)
        # init_weights(self.height_mat)

    def forward(self, x):
        # print(self.width_mat.min(), self.width_mat.max())
        # print(self.height_mat.min(), self.height_mat.max())
        # shift height and width dimensions to the left by one
        x = x.permute(0, 1, 3, 4, 2)
        rel_logits_w = self.relative_logits_width(x, self.width_mat)
        rel_logits_h = self.relative_logits_height(x, self.height_mat)

        return rel_logits_h, rel_logits_w

    def __rel_to_abs(self, x):
        """Converts tensor from relative to absolute indexing"""
        N, heads, L, _ = x.size()

        # padding in this way is necessary for proper elements
        # when reshaping at the end; i.e., padding in the obvious way
        # with torch.zeros((N, heads, 1, 2*L-1)) is incorrect.
        # To pad inner dimensions, this specific order must be followed
        # — one must pad from outer to inner, flattening in the process.
        col_pad = torch.zeros((N, heads, L, 1), device=x.device)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = x.flatten(start_dim=-2)
        flat_pad = torch.zeros((N, heads, L - 1), device=x.device)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
        final_x = flat_x_padded.view(N, heads, L + 1, 2 * L - 1)

        # why was it necessary to pad in order to recover the absolute positions?
        final_x = final_x[:, :, :L, L - 1:]
        return final_x

    def __relative_logits_1d(self, rel_logits, heads, size):
        rel_logits = rel_logits.view(-1, heads * size[0], size[1], 2 * size[1] - 1)
        rel_logits = self.__rel_to_abs(rel_logits)

        rel_logits = rel_logits.view(-1, heads, size[0], size[1], size[1])
        rel_logits = rel_logits.unsqueeze(dim=3)
        return rel_logits.repeat(1, 1, 1, size[1], 1, 1)

    def relative_logits_width(self, x, rel_k):
        N, heads, H, W, dk = x.size()
        rel_logits = x.matmul(rel_k.transpose(-1, -2))
        rel_logits = self.__relative_logits_1d(rel_logits, heads, (H, W))
        return rel_logits.transpose(3, 4).reshape(-1, heads, H * W, H * W)

    def relative_logits_height(self, x, rel_k):
        N, heads, H, W, dk = x.size()
        rel_logits = x.transpose(2, 3).matmul(rel_k.transpose(-1, -2))
        rel_logits = self.__relative_logits_1d(rel_logits, heads, (W, H))
        return rel_logits.permute(0, 1, 4, 2, 5, 3).reshape(-1, heads, H * W, H * W)


class AAConv2d(nn.Module):
    # (W−F+2P)/S+1
    def __init__(
            self, in_channels, out_channels, kernel_size, map_size,
            attn_size=(40, 40, 4), heads=4, stride=1, padding=0, bias=True, original=False
    ):
        super(AAConv2d, self).__init__()
        self.dq, self.dk, self.dv = attn_size
        self.heads = heads
        self.width = self.height = map_size

        assert self.dq == self.dk, 'AAConv2d requires dq == dk'
        assert heads > 0, 'AAConv2d requires a positive number of heads'
        assert type(kernel_size) == int, 'AAConv2d requires integer kernel_size'
        assert type(map_size) == int, 'AAConv2d requires integer map_size'
        assert self.dk % self.heads == 0, 'AAConv2d requires dk divisible by the number of heads'
        assert self.dv % self.heads == 0, 'AAConv2d requires dv divisible by the number of heads'
        assert stride in [1, 2], 'AAConv2d requires a stride of either 1 or 2'

        conv_out_channels = out_channels - self.dv
        attn_in_channels = attn_out_channels = self.dv
        self.conv = nn.Conv2d(in_channels, conv_out_channels, kernel_size, stride, padding, bias=bias)
        self.attn = nn.Conv2d(attn_in_channels, attn_out_channels, kernel_size=1, stride=1, bias=bias)
        self.relative_encoding = RelativeEncoding2d(map_size, map_size, self.dk // self.heads)
        self.q_conv = nn.Conv2d(in_channels, self.dk, kernel_size, stride, padding, bias=bias)
        self.k_conv = nn.Conv2d(in_channels, self.dk, kernel_size, stride, padding, bias=bias)
        self.v_conv = nn.Conv2d(in_channels, self.dv, kernel_size, stride, padding, bias=bias)
        self.temperature = self.dk // self.heads
        if original:
            padding = (kernel_size - 1) // 2
            self.conv = nn.Conv2d(in_channels, conv_out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
            self.attn = nn.Conv2d(attn_in_channels, attn_out_channels, kernel_size=1, stride=1, bias=bias)
            self.q_conv = nn.Conv2d(in_channels, self.dk, kernel_size=1, stride=1, bias=bias)
            self.k_conv = nn.Conv2d(in_channels, self.dk, kernel_size=1, stride=1, bias=bias)
            self.v_conv = nn.Conv2d(in_channels, self.dv, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        # A bit different than in the paper, where Q,K,V are generated with flattened maps and parameter matrices
        # TODO DEBUG matmul and einsum
        conv_maps = self.conv(x)
        N, C, H, W = conv_maps.size()
        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        # split attention heads
        q = q.view(N, self.heads, -1, H, W)
        k = k.view(N, self.heads, -1, H, W)
        v = v.view(N, self.heads, -1, H, W)

        logits = q.flatten(start_dim=-2).transpose(2, 3).matmul(k.flatten(start_dim=-2))
        h_rel_logits, w_rel_logits = self.relative_encoding(q)
        logits += h_rel_logits + w_rel_logits
        logits *= self.temperature ** -0.5
        weights = torch.softmax(logits, dim=-1)
        attn_maps = weights.matmul(v.flatten(start_dim=-2).transpose(2, 3))
        attn_maps = attn_maps.view(N, self.heads, -1, H, W)

        # combine attention heads
        attn_maps = attn_maps.view(N, -1, H, W)
        attn_maps = self.attn(attn_maps)
        return torch.cat((conv_maps, attn_maps), dim=1)


class AABaseline(nn.Module):
    def __init__(self, opts):
        super(AABaseline, self).__init__()
        # 32x32x3
        self.l1 = nn.Sequential(
            AAConv2d(in_channels=3, out_channels=256, kernel_size=5, heads=4, attn_size=(64, 64, 64),
                     map_size=14, stride=2, padding=0, original=False),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Dropout(p=0.1)
        )
        self.l1.apply(init_weights)

        # 7x7x16
        self.l2 = nn.Sequential(
            AAConv2d(in_channels=256, out_channels=512, kernel_size=4, heads=4, attn_size=(128, 128, 128),
                     map_size=4, stride=1, padding=0, original=False),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.1)
        )
        self.l2.apply(init_weights)

        # 2x2x32
        self.l3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=10, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.l3.apply(init_weights)

        # 1x1x10

    def forward(self, x):
        y = self.l3(self.l2(self.l1(x))).view(-1, 10)
        return y
