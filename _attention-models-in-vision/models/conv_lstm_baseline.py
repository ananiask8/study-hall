import torch
from torch import nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size=3, padding=1)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.zeros(state_size),
                torch.zeros(state_size)
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class UnrolledConvLSTMDecoderLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(UnrolledConvLSTMDecoderLayer, self).__init__()
        self.conv_lstm = ConvLSTMCell(input_size, hidden_size)

    def forward(self, x):
        features, state, idx, hidden = x
        (h, c) = state
        h, c = self.conv_lstm(features[idx], (h, c))
        return features, (h, c), idx + 1, hidden + [h]


class ConvLSTMDecoder(nn.Module):
    def __init__(self, input_size=3, hidden_size=512, patches_size=16):
        super(ConvLSTMDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.patches_size = patches_size
        self.unrolled_decoder = nn.Sequential(
            *[UnrolledConvLSTMDecoderLayer(input_size, hidden_size)] * patches_size
        )

    def forward(self, x):
        T, N, C, H, W = x.size()
        h = torch.zeros((N, self.hidden_size, H, W), device=x.device)
        c = torch.zeros((N, self.hidden_size, H, W), device=x.device)
        _, _, _, _, hidden = self.unrolled_decoder((x, (h, c), 0, []))
        return torch.cat(hidden, dim=1)


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


class ConvLSTMBaseline(nn.Module):
    def __init__(self):
        super(ConvLSTMBaseline, self).__init__()
        # Wout = (Win + 2P - D * (K - 1) - 1) / S + 1
        # padding = (kernel_size - 1) // 2 # for keeping the dimensions
        # Nx3x32x32
        self.backbone = nn.Sequential(
            TensorsByPatches(factor=4),
            ConvLSTMDecoder(input_size=3, hidden_size=256, patches_size=4*4)
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
