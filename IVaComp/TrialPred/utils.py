import torch
from torch.utils.data import Dataset
import numpy as np

# Create iva dataset
class IVaDataset(Dataset):

    def __init__(self, data_path, label_path):
        super(IVaDataset, self).__init__()
        self.data = torch.tensor(np.load(data_path, allow_pickle=True), dtype=torch.float32)
        self.label = torch.tensor(np.load(label_path, allow_pickle=True), dtype=torch.int32)
        self.num = np.shape(self.data)[0]

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        return self.data[item], self.label[item]

# # check: construct dataset for aa
# train_data_path = 'dataset/aa/train.npy'
# train_label_path = 'dataset/aa/train_label.npy'
# test_data_path = 'dataset/aa/test.npy'
# test_label_path = 'dataset/aa/test.npy'
# train_data = IVaDataset(train_data_path, train_label_path)
# test_data = IVaDataset(test_data_path, test_label_path)
# print(train_data.__getitem__(0))

