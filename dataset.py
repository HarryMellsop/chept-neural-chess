import random
import torch
from torch.utils.data import Dataset
import argparse
import math


class GamePredictionDataset(Dataset):

    def __init__(self, pretraining_dataset, data):

        self.MASK_CHAR = u"\u2047" # the doublequestionmark character, for mask
        self.PAD_CHAR = u"\u25A1" # the empty square character, for pad

        self.itos = pretraining_dataset.itos 
        self.stoi = pretraining_dataset.stoi 

        self.block_size = pretraining_dataset.block_size
        self.data = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        inp, oup = self.data[idx].split('\t')

        x = inp + self.MASK_CHAR + oup + self.MASK_CHAR
        x = x + self.PAD_CHAR * (self.block_size - len(x))
        y = self.PAD_CHAR * (len(inp) - 1) + x[len(inp):]
        
        x = x[:-1]

        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)

        return x, y