from torch.utils.data import Dataset


class BalancedDataset(Dataset):
    def __init__(self, dataset, max_per_class=50):
        self.data = []
        self.count = {y: 0 for x, y in dataset}
        for x, y in dataset:
            if self.count[y] < max_per_class:
                self.count[y] += 1
                self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
