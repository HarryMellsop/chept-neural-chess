import random
import torch
from torch.utils.data import Dataset
import argparse
import math


class PretrainDataset(Dataset):

    def __init__(self, data,
                 block_size=1024):
 
        self.block_size = block_size
        self.PAD_CHAR = u"\u25A1"

        chars = list(sorted(list(set(data))))
        assert self.PAD_CHAR not in chars
        chars.insert(0, self.PAD_CHAR)

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

        data_size, vocab_size = len(data), len(chars)
        print('Data has %d characters, %d unique.' % (data_size, vocab_size))

        self.data_size = data_size
        self.vocab_size = vocab_size
        
        self.data = data.split('\n')
        self.data = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        game = self.data[idx]
        game += self.PAD_CHAR * (self.block_size - len(game))

        x = game[:-1]
        y = game[1:]

        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)

        return x, y


if __name__ == '__main__':

    games = open('data/datasets-cleaned/kingbase_cleaned.txt').read()[:1000]
    print(games)
    # ds = PretrainDataset(games)