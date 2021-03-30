import torch
from torch import nn


class LabelSmoothingLoss(nn.Module):
    """
    Credits to Devin Yang (Devin.X.Y@outlook.com)
    """
    def __init__(self, smoothing=0.0, dim=-1, reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim
        self.reduction = reduction

    def forward(self, pred, target):
        n_cls = target.size(self.dim)
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        if self.reduction == 'none':
            return torch.sum(-true_dist * pred, dim=self.dim)
        elif self.reduction == 'mean':
            return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        elif self.reduction == 'sum':
            return torch.sum(torch.sum(-true_dist * pred, dim=self.dim))
