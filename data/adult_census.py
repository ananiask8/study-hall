import torch
import pandas as pd
from torch.utils.data import Dataset


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

    def __init__(self, path='adult.csv'):
        d = pd.read_csv(path)
        self.data = []
        for col in self.columns:
            self.data.append(torch.Tensor(self.map[col](d[col])))
        self.data = torch.stack(self.data, dim=1)

    def __index__(self, i):
        return self.data[i, :]
