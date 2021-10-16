import torch
from torch.utils.data import Dataset
import numpy as np

# Create Link prediction data set
class LPDataset(Dataset):

    def __init__(self, data_path, label_path):
        super(LPDataset, self).__init__()
        self.data = torch.from_numpy(np.load(data_path, allow_pickle=True))
        self.label = torch.from_numpy(np.load(label_path, allow_pickle=True))
        self.num = np.shape(self.data)[0]

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        return self.data[item], self.label[item]  # (5, 118, 118) (1, 118, 118)


def MSE(input, target):
    num = 1
    for s in input.size():
        num = num * s
    return (input - target).pow(2).sum().item() / num

def EdgeWiseKL(input, target):
    num = 1
    for s in input.size():
        num = num * s
    mask = (input > 0) & (target > 0)
    input = input.masked_select(mask)
    target = target.masked_select(mask)
    kl = (target * torch.log(target / input)).sum().item() / num
    return kl

def MissRate(input, target):
    num = 1
    for s in input.size():
        num = num * s
    mask1 = (input > 0) & (target == 0)
    mask2 = (input == 0) & (target > 0)
    mask = mask1 | mask2
    return mask.sum().item() / num