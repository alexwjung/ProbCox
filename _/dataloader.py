import numpy as np
import pandas as pd
import torch
from  torch.utils.data import Dataset

def custom_collate(batch):
    surv = torch.cat([item[0] for item in batch], 0)
    X = torch.cat([item[1] for item in batch], 0)
    return((surv, X))

class ToTensor(object):
    '''Convert ndarrays in sample to Tensors.'''
    def __call__(self, sample):
        return [torch.from_numpy(sample[i]) for i in range(len(sample))]

class dataloader(Dataset):
    """Dataset."""

    def __init__(self, surv, X, transform=None):
        """
        Args:
            
        """
        self.surv = surv
        self.X = X
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, ii):

        sample = (self.surv[ii, :][None, :], self.X[ii, :][None, :])

        if self.transform:
            sample = self.transform(sample)

        return sample

class CSV_Dataset():
    """Dataset."""

    def __init__(self, path_data, batch_size=1, transform=None):
        """
        Args:
            path_data (string): Path to the csv file
        """
        self.path_data = path_data
        self.data = pd.read_csv(path_data, sep=';', iterator=True, chunksize=batch_size)
        self.transform = transform

    def __len__(self):
        #with open(self.path_data) as f:
        #    ll = sum(1 for line in f)
        return 1000000000000000000

    def __getitem__(self, ii):
        d = self.data.get_chunk()
        surv = np.asarray(d.iloc[:, :3]).astype(float)
        X = np.asarray(d.iloc[:, 3:]).astype(float)
        if self.transform:
            (surv, X) = self.transform((surv, X))
        return (surv, X)
        