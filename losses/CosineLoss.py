import torch
from torch import nn


class CosineLoss(nn.Module):
    def __init__(self, eps=1e-6, one_hot=True, reduction='none'):
        super(CosineLoss, self).__init__()
        self.eps = eps
        self.one_hot = one_hot
        self.reduction = reduction

    def forward(self, y, target):
        if not self.one_hot:
            target = target.squeeze(-1).unsqueeze(-1)
            target = torch.scatter(
                torch.zeros_like(y, dtype=torch.float),
                -1,
                target,
                torch.ones_like(target, dtype=torch.float))
        y = y / y.norm(p=2, dim=-1, keepdim=True).clamp(min=self.eps)
        loss = 1. - torch.einsum('...i,...i->...', target, y)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
