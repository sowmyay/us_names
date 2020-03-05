import string

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


from ngRNN.transforms import (
    StateToTensor,
    NameToTensor,
    TargetToTensor
)


class RGDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.names = df
        all_letters = string.ascii_letters
        all_states = self.names.state.unique().tolist()

        self.state_transform = StateToTensor(all_states)
        self.name_transform = NameToTensor(all_letters)
        self.target_transform = TargetToTensor(all_letters)
        self.max_len = 15

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        row = self.names.iloc[i]
        state_tensor = self.state_transform(row["state"])
        name_tensor = self.name_transform(row["name"])
        s = name_tensor.size()
        name_tensor = torch.cat([name_tensor, torch.zeros(self.max_len-s[0], s[1], s[2])], dim=0)
        target_tensor = self.target_transform(row["name"])
        s = target_tensor.size()
        target_tensor = torch.cat([target_tensor, torch.zeros(self.max_len - s[0], dtype=torch.long)], dim=0)
        return state_tensor, name_tensor, target_tensor

    def n_states(self):
        return len(self.names.state.unique())