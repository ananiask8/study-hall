from torch import nn
import operator
from functools import reduce
from einops.layers.torch import Rearrange


class ImageAutoencoder(nn.Module):
    def __init__(self, size=(3, 32, 32), scale=2, layers=3):
        super(ImageAutoencoder, self).__init__()
        self.size = size
        feature_size = reduce(operator.mul, size, 1)
        self.encoder = nn.Sequential(
            Rearrange('N C H W -> N (C H W)'),
            *[nn.Sequential(
                nn.Linear(feature_size // (i * scale), feature_size // ((i + 1) * scale)),
                nn.ReLU()
            ) for i in range(1, layers + 1)]
        )
        self.decoder = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(feature_size // ((i + 1) * scale), feature_size // (i * scale)),
                nn.ReLU()
            ) for i in reversed(range(1, layers))],
            nn.Linear(feature_size // scale, feature_size),
            Rearrange('N (C H W) -> N C H W', C=size[0], H=size[1], W=size[2])
        )

    def forward(self, x):
        representation = self.encoder(x)
        return self.decoder(x), representation
