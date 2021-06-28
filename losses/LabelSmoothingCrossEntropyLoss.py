import torch
from torch import nn
from torch.nn import functional as F


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """
    Credits to seominseok0429
    """
    def __init__(self, smoothing=0.1, reduction='none'):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
