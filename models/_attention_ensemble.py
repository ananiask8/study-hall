import torch
from torch import nn
from torch.nn import functional as F


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class EnsemblePredictor(nn.Module):
    def __init__(self, features=(0, 1, 2), embeddings=(None, 8, 4)):
        super(EnsemblePredictor, self).__init__()
        self.fidx = features
        self.embeddings = nn.ModuleList(
            (nn.Embedding(2*e, e) if e is not None else lambda x: x)
            for e in embeddings
        )
        in_features = sum([(e if e is not None else 1) for e in embeddings])
        self.f = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
        )
        self.f.apply(init_weights)
        self.classifier = nn.Linear(256, 1)
        self.classifier.apply(init_weights)

    def forward(self, x):
        e = torch.cat([self.embeddings[i](x[:, fidx]) for i, fidx in enumerate(self.fidx)], dim=1)
        features = self.f(e)
        return self.classifier(features), features


class BaselinePredictor(nn.Module):
    def __init__(self, embeddings=(None, 5, None, 8, None, 4, 8, 3, 3, 1, None, None, None, 21)):
        super(BaselinePredictor, self).__init__()
        self.embeddings = nn.ModuleList(
            (nn.Embedding(2*e, e) if e is not None else lambda x: x)
            for e in embeddings
        )
        in_features = sum((e if e is not None else 1) for e in embeddings)
        self.f = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
        )
        self.f.apply(init_weights)
        self.classifier = nn.Linear(256, 1)
        self.classifier.apply(init_weights)

    def forward(self, x):
        e = torch.cat([self.embeddings[i](x[:, i]) for i in range(x.size(1))], dim=1)
        features = self.f(e)
        return self.classifier(features), features


class TransformerStackedEnsemble(nn.Module):
    def __init__(self):
        super(TransformerStackedEnsemble, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(256, 4, 256, 0.1)
        encoder_norm = nn.LayerNorm(256)
        self.encoder = nn.TransformerEncoder(encoder_layers, 4, encoder_norm)
        self.encoder.apply(init_weights)
        self.f = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
        self.f.apply(init_weights)
        self.classifier = lambda x: F.softmax(x, dim=0)

    def forward(self, x, px):
        memory = self.encoder(x)
        return torch.einsum('kn,kn->n', px, self.classifier(self.f(memory)))


class MHAStackedEnsemble(nn.Module):
    def __init__(self):
        super(MHAStackedEnsemble, self).__init__()
        self.mha = nn.MultiheadAttention(256, 4)
        init_weights(self.mha)

    def forward(self, x, px):
        _, attn_weights = self.mha(query=x, key=x, value=x, need_weights=True)
        return torch.einsum('kn,kn->n', px, attn_weights)


class TransformerFeatureMixerEnsemble(nn.Module):
    def __init__(self):
        super(TransformerFeatureMixerEnsemble, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(256, 4, 256, 0.1)
        encoder_norm = nn.LayerNorm(256)
        self.encoder = nn.TransformerEncoder(encoder_layers, 4, encoder_norm)
        self.encoder.apply(init_weights)
        self.f = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
        self.f.apply(init_weights)
        self.classifier = lambda x: self.f(x.mean(dim=0))

    def forward(self, x, px):
        memory = self.encoder(x)
        return self.classifier(memory)
