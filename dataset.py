import random
import torch
from torch.utils.data import Dataset


class Finetune_Full(Dataset):

    def __init__(self, data, block_size, pretrain_vocab):

        assert pretrain_vocab, "Must have pretrain vocab for finetuning"

        self.block_size = block_size
        self.PAD_CHAR = u"\u25A1"
        self.MASK_CHAR = u"\u2047"

        self.stoi = pretrain_vocab['stoi']
        self.itos = pretrain_vocab['itos']

        assert len(self.stoi) == len(self.itos)

        self.vocab_size = len(self.stoi)
        self.data_size = len(data)

        print('Data has %d characters, %d unique.' % (self.data_size, self.vocab_size))

        self.data = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))[:-1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        game = self.data[idx]
        spaces = [idx for idx, cur in enumerate(game) if cur == ' ']
        n_spaces = len(spaces)

        # index = random.randint((n_spaces // 2) - 1, n_spaces - 2)
        # doesn't know what to do early game now cuz it was never "prompted"
        # finetuning #2 could be to focus only on ^^ >= 50% of game

        # different way to calc loss? 99% of it is just padding 512 chars
        index = random.randint(0, n_spaces - 2)
        m_start, m_stop = spaces[index] + 1, spaces[index + 1]
        x = game[:m_start] + self.MASK_CHAR + game[m_start:m_stop + 1] + self.MASK_CHAR
        x = x + self.PAD_CHAR * (self.block_size - len(x))
        y = self.PAD_CHAR * m_start + self.MASK_CHAR + game[m_start:m_stop + 1] + self.MASK_CHAR
        y = y + self.PAD_CHAR * (self.block_size - len(y))

        assert len(x) == len(y) == self.block_size

        x = x[:-1]
        y = y[1:]

        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)

        return x, y


class Finetune_Middle(Dataset):

    def __init__(self, data, block_size, pretrain_vocab):

        assert pretrain_vocab, "Must have pretrain vocab for finetuning"

        self.block_size = block_size
        self.PAD_CHAR = u"\u25A1"
        self.MASK_CHAR = u"\u2047"

        self.stoi = pretrain_vocab['stoi']
        self.itos = pretrain_vocab['itos']

        assert len(self.stoi) == len(self.itos)

        self.vocab_size = len(self.stoi)
        self.data_size = len(data)

        print('Data has %d characters, %d unique.' % (self.data_size, self.vocab_size))

        self.data = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))[:-1]
        self.starting = int(0.2 * len(self.data))
        self.data = self.data[self.starting:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        game = self.data[idx]
        spaces = [idx for idx, cur in enumerate(game) if cur == ' ']
        n_spaces = len(spaces)

        index = random.randint((n_spaces // 2) - 1, n_spaces - 2)
        m_start, m_stop = spaces[index] + 1, spaces[index + 1]
        x = game[:m_start] + self.MASK_CHAR + game[m_start:m_stop + 1] + self.MASK_CHAR
        x = x + self.PAD_CHAR * (self.block_size - len(x))
        y = self.PAD_CHAR * m_start + self.MASK_CHAR + game[m_start:m_stop + 1] + self.MASK_CHAR
        y = y + self.PAD_CHAR * (self.block_size - len(y))

        assert len(x) == len(y) == self.block_size

        x = x[:-1]
        y = y[1:]

        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)

        return x, y


class Directory:

    def __init__(self, data, version, config_args, pretrain_vocab=None):

        self.data = data
        self.version = version
        self.config_args = config_args
        self.pretrain_vocab = pretrain_vocab

        self.direct = {None: PretrainDataset,
                       0: Finetune_Full,
                       1: Finetune_Middle}

    def __call__(self):

        return self.direct[self.version](self.data, self.config_args['block_size'], self.pretrain_vocab)


class ChessMoveDataset(Dataset):
    def __init__(self, data, block_size=1024, pretrain_vocab=None):

        self.block_size = block_size
        self.PAD_CHAR = u"\u25A1"
        self.MASK_CHAR = u"\u2047"
        chars = list(sorted(list(set(data))))
        if '\n' in chars:
            chars.remove('\n')

        # Check and insert pad and mask chars
        assert self.PAD_CHAR not in chars, 'Pad character redundant!'
        chars.insert(0, self.PAD_CHAR)
        assert self.MASK_CHAR not in chars, 'Mask character redundant!'
        chars.insert(0, self.MASK_CHAR)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        data_size, vocab_size = len(data), len(chars)
        print('Data has %d characters, %d unique.' % (data_size, vocab_size))

        self.data_size = data_size
        self.vocab_size = vocab_size

        self.data = list(data.encode('utf-8').decode('ascii', errors='ignore'))

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

    def __init__(self, data, block_size=1024, pretrain_vocab=None):

        self.block_size = block_size
        self.PAD_CHAR = u"\u25A1"
        self.MASK_CHAR = u"\u2047"

        moves = list(set(data))

        # Check and insert pad and mask chars
        assert self.PAD_CHAR not in moves, 'Pad character redundant!'
        moves.insert(0, self.PAD_CHAR)
        assert self.MASK_CHAR not in moves, 'Mask character redundant!'
        moves.insert(0, self.MASK_CHAR)

        self.stoi = {ch: i for i, ch in enumerate(moves)}
        self.itos = {i: ch for i, ch in enumerate(moves)}

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
                 block_size=1024,
                 pretrain_vocab=None):

        self.block_size = block_size
        self.PAD_CHAR = u"\u25A1"
        self.MASK_CHAR = u"\u2047"

        chars = list(sorted(list(set(data))))
        if '\n' in chars:
            chars.remove('\n')

        # Check and insert pad and mask chars
        assert self.PAD_CHAR not in chars, 'Pad character redundant!'
        chars.insert(0, self.PAD_CHAR)
        assert self.MASK_CHAR not in chars, 'Mask character redundant!'
        chars.insert(0, self.MASK_CHAR)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        data_size, vocab_size = len(data), len(chars)
        print('Data has %d characters, %d unique.' % (data_size, vocab_size))

        self.data_size = data_size
        self.vocab_size = vocab_size

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

    finetune_versions = {0: Finetune_Full,
                         1: Finetune_Middle}


if __name__ == '__main__':

    games = open('data/datasets-cleaned/kingbase_cleaned.txt').read()[:1000]
    print(games)
    # ds = PretrainDataset(games)
