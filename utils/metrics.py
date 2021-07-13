import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from typing import NamedTuple, Optional, Tuple, List
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve


class Metrics(NamedTuple):
    accuracy: Optional[float]
    f1score: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    roc_auc: Optional[float]
    roc_curve: Optional[Tuple[List[float], List[float], List[float]]]
    pr_auc: Optional[float]
    pr_curve: Optional[Tuple[List[float], List[float], List[float]]]


def performance(preds: ndarray, targets: ndarray,
                threshold: float = 0.5, calculate_curves: bool = False) -> Metrics:
    if preds.size == 0 or targets.size == 0 or preds.shape != targets.shape:
        raise AttributeError('Invalid pair of predictions and targets in performance computation.')

    hard_preds = (preds > threshold)
    hard_targets = (targets > threshold)
    tn, fp, fn, tp = confusion_matrix(hard_targets, hard_preds, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp > 0 else float('inf')
    recall = tp / (tp + fn) if tp + fn > 0 else float('inf')
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1score = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else None
    roc_auc_var, roc_curve_var, pr_curve_var, pr_auc_var = None, None, None, None
    if calculate_curves:
        roc_curve_var = roc_curve(hard_targets, preds)
        roc_auc_var = roc_auc_score(hard_targets, preds)
        pr_curve_var = precisions, recalls, thresholds = precision_recall_curve(hard_targets, preds)
        pr_auc_var = auc(recalls, precisions)
    return Metrics(
        accuracy=acc,
        f1score=f1score,
        precision=precision,
        recall=recall,
        roc_auc=roc_auc_var,
        roc_curve=roc_curve_var,
        pr_auc=pr_auc_var,
        pr_curve=pr_curve_var
    )


class F1Score(nn.Module):
    """
    Calculate F1 score. Can work with gpu tensors.

    The original implementation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    """

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


def evaluate_calibration(preds: ndarray, targets: ndarray, name: str = ''):
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    fraction_of_positives, mean_predicted_value = calibration_curve(targets, preds, n_bins=10)
    ax1.plot(mean_predicted_value, fraction_of_positives, 's-', label=f'{name}')
    ax2.hist(preds, range=(0, 1), bins=10, label=name, histtype='step', lw=2)

    calibrated = np.arange(0, 1.1, step=0.1)
    ax1.plot(calibrated, calibrated, color='black', linestyle='dashed')

    ax1.set_ylabel('Fraction of positives')
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc='lower right')
    ax1.set_title('Calibration plots (reliability curve)')

    ax2.set_xlabel('Mean predicted value')
    ax2.set_ylabel('Count')
    ax2.legend(loc='upper center', ncol=2)

    plt.tight_layout()
    return fig
