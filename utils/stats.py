import torch
from typing import NamedTuple


class DatasetStats(NamedTuple):
    mean: torch.Tensor
    std: torch.Tensor
    class_weights: torch.Tensor


def get_dataset_stats(data):
    all_x, all_y = [], []
    for x, y in data:
        all_x.append(x)
        all_y.append(torch.Tensor([y]))
    all_x, all_y = torch.stack(all_x, dim=0).float().flatten(start_dim=2), torch.stack(all_y, dim=0).long().view(-1)
    mean, std = all_x.mean(dim=0), all_x.std(dim=0)
    classes = torch.bincount(all_y).float()
    classes /= classes.sum()
    if classes.size(0) == 2:
        classes = classes[0] / classes[1]
    return DatasetStats(mean, std, classes)
