import random
import torch
from torch.utils.data import Dataset
import argparse
import math

class ChessMoveDataset(Dataset):
    def __init__(self, data, block_size=1024):

        self.block_size = block_size
        self.PAD_CHAR = u"\u25A1"
        self.MASK_CHAR = u"\u2047"
        chars = list(sorted(list(set(data))))
        assert self.PAD_CHAR not in chars
        chars.insert(0, self.PAD_CHAR)

        assert self.MASK_CHAR not in chars
        chars.insert(0, self.MASK_CHAR)
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
        # randomly select a chess move to remove and have the model predict

        game = self.data[idx]

        moves = [move for move in game.split()]
        
        index = random.randint(0, len(moves) - 1)
        masked_move = moves[index]
        
        if index % 2 == 0:
            masked_move = masked_move[2:]
            moves[index] = moves[index][:2]
        else:
            moves[index] = ""

        # now, we have masked_move as the actual masked move, and the moves list has been left as if only the move itself were removed

        prefix = " ".join(moves[:index + 1])
        suffix = " ".join(moves[index + 1:])

        masked_string = prefix + self.MASK_CHAR + " " + suffix + self.MASK_CHAR + masked_move + self.MASK_CHAR

        num_pad_chars = self.block_size - len(masked_string)
        masked_string += num_pad_chars * self.PAD_CHAR

        input = masked_string[:-1]
        output = masked_string[1:]

        x = torch.tensor([self.stoi[c] for c in input], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in output], dtype=torch.long)
        return x, y

class FullEmbeddingPretrainDataset(Dataset):

    # requires that we clean numbers out of training data
    # generate every possible PGN move, store that into 
    
    def __init__(self, data, block_size=1024):

        self.block_size = block_size
        self.PAD_CHAR = u"\u25A1"

        moves = list(set(data.split()))
        assert self.PAD_CHAR not in moves
        moves.insert(0, self.PAD_CHAR)

        self.stoi = { ch:i for i,ch in enumerate(moves) }
        self.itos = { i:ch for i,ch in enumerate(moves) }

        data_size, vocab_size = len(data), len(moves)
        print('Data has %d games, %d unique moves.' % (data_size, vocab_size))

        self.data_size = data_size
        self.vocab_size = vocab_size
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        game = self.data[idx].split()
        game += self.PAD_CHAR * (self.block_size - len(game))

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
