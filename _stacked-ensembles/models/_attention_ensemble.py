import torch
from torch import nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class EnsembleIndependentPredictor(nn.Module):
    def __init__(self, features=(0, 1, 2), embeddings=(None, 8, 4)):
        super(EnsembleIndependentPredictor, self).__init__()
        self.fidx = features
        self.embeddings = nn.ModuleList(
            (nn.Embedding(2*e, e) if e and e > 0 else nn.BatchNorm1d(1, affine=False))
            for e in embeddings
        )
        in_features = sum([(e if e and e > 0 else 1) for e in embeddings])
        self.f = nn.Sequential(
            nn.Linear(in_features, 32, bias=False),
            nn.Tanh(),
            nn.BatchNorm1d(32, affine=False),
            nn.Linear(32, 16, bias=False),
            nn.Tanh()
        )
        self.f.apply(init_weights)
        self.classifier = nn.Linear(16, 1, bias=False)
        self.classifier.apply(init_weights)

    def forward(self, x):
        N = x.size(0)
        e = []
        for i, fid in enumerate(self.fidx):
            x_in = x[:, fid] if isinstance(self.embeddings[i], nn.Embedding) else x[:, fid].float().unsqueeze(-1)
            e.append(self.embeddings[i](x_in).float().view(N, -1))
        e = torch.cat(e, dim=1)
        features = self.f(e)
        return self.classifier(features), features


class BaselinePredictor(nn.Module):
    """
    Baseline class to compare the TransformerStackedEnsemble to
    Note: embeddings[i] = None — uses the feature as is rather than as an embedding
    """
    def __init__(self, embeddings=(None, 5, None, 8, None, 4, 8, 3, 3, 1, None, None, None, 21)):
        super(BaselinePredictor, self).__init__()
        self.embeddings = nn.ModuleList(
            (nn.Embedding(2*e, e) if e is not None else nn.BatchNorm1d(1, affine=False))
            for e in embeddings
        )
        in_features = sum((e if e is not None else 1) for e in embeddings)
        self.f = nn.Sequential(
            nn.Linear(in_features, 32, bias=False),
            nn.Tanh(),
            nn.BatchNorm1d(32, affine=False),
            nn.Linear(32, 16, bias=False),
            nn.Tanh()
        )
        self.f.apply(init_weights)
        self.classifier = nn.Sequential(
            nn.Linear(16, 1, bias=False)
        )
        self.classifier.apply(init_weights)

    def forward(self, x):
        N = x.size(0)
        e = []
        for i in range(x.size(1)):
            x_in = x[:, i] if isinstance(self.embeddings[i], nn.Embedding) else x[:, i].float().unsqueeze(-1)
            e.append(self.embeddings[i](x_in).float().view(N, -1))
        e = torch.cat(e, dim=1)
        features = self.f(e)
        return self.classifier(features), features


class BaselineTransformer(nn.Module):
    """
    Baseline class to compare the TransformerStackedEnsemble to
    Note: embeddings[i] = None — uses the feature as is rather than as an embedding
    """
    def __init__(self, embeddings=(None, 10, None, 16, None, 8, 16, 6, 6, 2, None, None, None, 42)):
        super(BaselineTransformer, self).__init__()
        sz = 16
        self.embeddings = nn.ModuleList(
            (nn.Embedding(e, sz) if e is not None else nn.Sequential(
                nn.Linear(1, sz, bias=False),
                nn.Tanh(),
                nn.BatchNorm1d(sz, affine=False),
            ) for e in embeddings)
        )
        self.te = nn.TransformerEncoderLayer(sz, 1, dim_feedforward=sz, dropout=0.1, activation='relu')
        self.ln = nn.LayerNorm(sz)
        self.encoder = nn.TransformerEncoder(self.te, num_layers=2, norm=self.ln)
        init_weights(self.encoder)
        cat_size = sz*len(embeddings)
        self.classifier = nn.Sequential(
            nn.Linear(cat_size, 1, bias=False)
            # nn.Linear(sz, 1, bias=False)
        )
        self.classifier.apply(init_weights)

    def forward(self, x):
        N = x.size(0)
        e = []
        for i in range(x.size(1)):
            x_in = x[:, i] if isinstance(self.embeddings[i], nn.Embedding) else x[:, i].float().unsqueeze(-1)
            e.append(self.embeddings[i](x_in).float().view(N, -1))
        features = torch.stack(e, dim=0)
        values = self.encoder(features)
        attn_values = torch.flatten(values.permute(1, 2, 0), start_dim=1, end_dim=2)
        # attn_values = values.permute(1, 2, 0).mean(-1)
        return self.classifier(attn_values), attn_values


class BaseStackedEnsemble(nn.Module):
    def __init__(self):
        super(BaseStackedEnsemble, self).__init__()
        self.components = nn.ModuleList([
            EnsembleIndependentPredictor(features=(1, 3, 5), embeddings=(5, 8, 4)),
            EnsembleIndependentPredictor(features=(6, 7, 8, 9), embeddings=(8, 3, 3, 1)),
            EnsembleIndependentPredictor(
                features=(0, 2, 4, 10, 11, 12, 13), embeddings=(None, None, None, None, None, None, 21))
        ])
        self.components[0].load_state_dict(torch.load('resources/ip1.pth', map_location='cpu')['model_state_dict'])
        self.components[1].load_state_dict(torch.load('resources/ip2.pth', map_location='cpu')['model_state_dict'])
        self.components[2].load_state_dict(torch.load('resources/ip3.pth', map_location='cpu')['model_state_dict'])

    def components_eval_predict(self, x, dim=0):
        preds, features = [], []
        for f in self.components:
            f.eval()
            with torch.no_grad():
                pred, feature = f(x)
            preds.append(pred)
            features.append(feature)
        preds = torch.stack(preds, dim)
        features = torch.stack(features, dim)
        return preds, features

    def components_train_predict(self, x, dim=0):
        preds, features = [], []
        for f in self.components:
            f.train()
            pred, feature = f(x)
            preds.append(pred)
            features.append(feature)
        preds = torch.stack(preds, dim)
        features = torch.stack(features, dim)
        return preds, features


class MeanStackedEnsemble(BaseStackedEnsemble):
    def __init__(self):
        super(MeanStackedEnsemble, self).__init__()
        self.stub = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        preds, features = self.components_eval_predict(x, dim=0)
        y = preds.squeeze(-1).mean(0)
        return y + 0*self.stub, features


class MHAStackedEnsemble(BaseStackedEnsemble):
    def __init__(self):
        super(MHAStackedEnsemble, self).__init__()
        self.mha = nn.MultiheadAttention(17, 1)
        init_weights(self.mha)
        self.classifier = nn.Sequential(
            nn.Linear(3*17, 1),
        )
        init_weights(self.classifier)

    def forward(self, x):
        preds, features = self.components_eval_predict(x, dim=0)
        features = torch.cat((preds, features), dim=-1)
        attn_values = self.mha(query=features, key=features, value=features, need_weights=False)[0]
        attn_values = torch.flatten(attn_values.permute(1, 2, 0), start_dim=1, end_dim=2)
        return self.classifier(attn_values), attn_values


class MHAConvexCombinationStackedEnsemble(BaseStackedEnsemble):
    def __init__(self):
        super(MHAConvexCombinationStackedEnsemble, self).__init__()
        self.mha = nn.MultiheadAttention(16, 1)

    def forward(self, x):
        preds, features = self.components_eval_predict(x, dim=0)
        preds = preds.squeeze(-1)
        _, attn_weights = self.mha(query=features, key=features, value=features, need_weights=True)
        return torch.einsum('sn,nls->nl', preds, attn_weights).mean(-1), features


class NaiveConvexCombinationStackedEnsemble(BaseStackedEnsemble):
    def __init__(self):
        super(NaiveConvexCombinationStackedEnsemble, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(3*16, int(1.5*16)),
            nn.Tanh(),
            nn.Linear(int(1.5*16), 3),
            nn.Softmax(dim=1)
        )
        init_weights(self.f)

    def forward(self, x):
        preds, features = self.components_eval_predict(x, dim=-1)
        preds = torch.flatten(preds, start_dim=-2, end_dim=-1)
        features = torch.flatten(features, start_dim=-2, end_dim=-1)
        attn_weights = self.f(features)
        return torch.einsum('ns,ns->n', preds, attn_weights), features


class NaiveStackedEnsemble(BaseStackedEnsemble):
    def __init__(self):
        super(NaiveStackedEnsemble, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(3*17, int(1.5*17)),
            nn.Tanh(),
            nn.Linear(int(1.5*17), 1)
        )
        init_weights(self.f)

    def forward(self, x):
        preds, features = self.components_eval_predict(x, dim=-1)
        features = torch.cat((preds, features), dim=-2)
        features = torch.flatten(features, start_dim=-2, end_dim=-1)
        attn_weights = self.f(features)
        return attn_weights, features


class TransformerBasedStackedEnsemble(BaseStackedEnsemble):
    def __init__(self):
        super(TransformerBasedStackedEnsemble, self).__init__()
        self.te = nn.TransformerEncoderLayer(17, 1, dim_feedforward=17, dropout=0.1, activation='relu')
        self.ln = nn.LayerNorm(17)
        self.encoder = nn.TransformerEncoder(self.te, num_layers=1, norm=self.ln)
        init_weights(self.encoder)
        self.classifier = nn.Sequential(
            nn.Linear(3*17, 1),
        )
        init_weights(self.classifier)

    def forward(self, x):
        preds, features = self.components_eval_predict(x, dim=0)
        features = torch.cat((preds, features), dim=-1)
        values = self.encoder(features)
        attn_values = torch.flatten(values.permute(1, 2, 0), start_dim=1, end_dim=2)
        return self.classifier(attn_values), features


class TransformerBasedRandomizedStackedEnsemble(nn.Module):
    embeddings = torch.IntTensor([-1, 5, -1, 8, -1, 4, 8, 3, 3, 1, -1, -1, -1, 21])

    def __init__(self, n_components=32, n_features_per_component=4):
        super(TransformerBasedRandomizedStackedEnsemble, self).__init__()
        torch.manual_seed(42)
        self.idx = torch.randint(self.embeddings.size(0), (n_components, n_features_per_component))
        self.components = nn.ModuleList([
            EnsembleIndependentPredictor(
                features=self.idx[i, :].tolist(), embeddings=self.embeddings[self.idx[i, :]].tolist()
            )
            for i in range(n_components)
        ])
        self.te = nn.TransformerEncoderLayer(17, 1, dim_feedforward=17, dropout=0, activation='relu')
        self.ln = nn.LayerNorm(17)
        self.encoder = nn.TransformerEncoder(self.te, num_layers=1, norm=self.ln)
        init_weights(self.encoder)
        self.classifier = nn.Sequential(
            nn.Linear(n_components*17, 1),
        )
        init_weights(self.classifier)

    def forward(self, x):
        preds, features = [], []
        for f in self.components:
            pred, feature = f(x)
            preds.append(pred)
            features.append(feature)
        preds = torch.stack(preds, dim=0)
        features = torch.stack(features, dim=0)
        features = torch.cat((preds, features), dim=-1)
        values = self.encoder(features)
        attn_values = torch.flatten(values.permute(1, 2, 0), start_dim=1, end_dim=2)
        return self.classifier(attn_values), features
