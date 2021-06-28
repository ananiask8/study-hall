import torch
import pandas as pd
from torch.utils.data import Dataset

from utils.stats import DatasetStats


class AdultCensusDataset(Dataset):
    # use loss with weights; neg 76%, pos 24%
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education.num',
        'marital.status', 'occupation', 'relationship', 'race', 'sex',
        'capital.gain', 'capital.loss', 'hours.per.week', 'native.country',
        'income'
    ]
    map = {
        'age': lambda x: x.values,
        'workclass': lambda x: x.astype('category').cat.codes.values,  # embed in 5
        'fnlwgt': lambda x: x.values,
        'education': lambda x: x.astype('category').cat.codes.values,  # embed in 8
        'education.num': lambda x: x.values,
        'marital.status': lambda x: x.astype('category').cat.codes.values,  # embed in 4
        'occupation': lambda x: x.astype('category').cat.codes.values,  # embed in 8
        'relationship': lambda x: x.astype('category').cat.codes.values,  # embed in 3
        'race': lambda x: x.astype('category').cat.codes.values,  # embed in 3
        'sex': lambda x: x.astype('category').cat.codes.values,  # embed in 1
        'capital.gain': lambda x: x.values,
        'capital.loss': lambda x: x.values,
        'hours.per.week': lambda x: x.values,
        'native.country': lambda x: x.astype('category').cat.codes.values,  # embed in 21
        'income': lambda x: x.astype('category').cat.codes.values
    }
    label = 'income'

    def __init__(self, path='data/adult.csv'):
        d = pd.read_csv(path)
        self.data = []
        for col in self.columns:
            self.data.append(torch.LongTensor(self.map[col](d[col])))
        label_idx = self.columns.index(self.label)
        self.data = torch.stack(self.data, dim=1)
        x = torch.cat([self.data[:, :label_idx], self.data[:, label_idx + 1:]], dim=1)
        y = self.data[:, label_idx].unsqueeze(-1).float()
        self.data = list(zip(*[x, y]))

    def __getitem__(self, i):
        return self.data[i]
