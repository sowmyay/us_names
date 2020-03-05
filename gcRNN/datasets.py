import string

import torch
from torch.utils.data import Dataset

from gcRNN.transforms import (
    GenderToTensor,
    NameToTensor
)


class GCDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.names = df
        all_letters = string.ascii_letters

        self.gender_transform = GenderToTensor()
        self.name_transform = NameToTensor(all_letters)
        self.max_len = 15

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        row = self.names.iloc[i]
        state_tensor = self.gender_transform(row["gender"])
        name_tensor = self.name_transform(row["name"])
        s = name_tensor.size()
        name_tensor = torch.cat([name_tensor, torch.zeros(self.max_len-s[0], s[1], s[2])], dim=0)
        return state_tensor, name_tensor


class GCPredictDataset(Dataset):
    def __init__(self):
        super().__init__()
        all_letters = string.ascii_letters
        self.name_transform = NameToTensor(all_letters)
        self.max_len = 15

    def __getitem__(self, name):
        name_tensor = self.name_transform(name)
        s = name_tensor.size()
        name_tensor = torch.cat([name_tensor, torch.zeros(self.max_len-s[0], s[1], s[2])], dim=0)
        return name_tensor
