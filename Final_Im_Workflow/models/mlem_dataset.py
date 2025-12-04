import torch
from torch.utils.data import Dataset

class MLEMDataset(Dataset):
    def __init__(self, input, target):
        self.input = input
        self.target = target

    def __len__(self):
        return(len(self.target))

    #Enables us to get one item at a time
    def __getitem__(self, idx):
        inp = torch.from_numpy(self.input[idx]).float().unsqueeze(0)
        tar = torch.from_numpy(self.target[idx]).float().unsqueeze(0)
        return inp, tar