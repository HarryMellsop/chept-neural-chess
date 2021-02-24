import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
import random
import dataset
import model
import trainer
import utils
import dataset
import pickle

# hardcode hyperparams
vocab_size = 37
block_size = 512

# save the device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

games = open('data/datasets-cleaned/kingbase_cleaned.txt').read()
games = games[:int(len(games) / 4)]
pretrain_dataset = dataset.PretrainDataset(games, block_size=block_size)

print(len(pretrain_dataset.stoi))
print(len(pretrain_dataset.itos))

# build model config
mconf = model.GPTConfig(
    vocab_size=vocab_size, 
    block_size=block_size, 
    n_layer=32, 
    n_head=8, 
    n_embd=256
)

# load model weights
model = model.GPT(mconf)
model.load_state_dict(torch.load('ckpt/model.iter.params', map_location=torch.device('cpu')))

# load dataset
with open('cache/stoi.pkl', 'rb') as f: 
    stoi = pickle.load(f)
    print(len(stoi))
with open('cache/itos.pkl', 'rb') as f:
    itos = pickle.load(f)
    print(len(itos))

def get_prediction(game_str):

    x = game_str
    x = torch.tensor([stoi[s] for s in x], dtype=torch.long)
    x = x.view(1, -1)

    model.eval()
    with torch.no_grad():

        logits, _ = model(x)
        logits = torch.squeeze(logits)
        y_hat = torch.argmax(logits, dim=-1)
        y_hat = [itos[t.item()] for t in y_hat]

    pred = y_hat[len(game_str) - 1]
    return pred

# run inference loop
game_str = ''
bot_move = ''
print('Welcome to Chess Bot. Enter moves below to start a game.')

while True:
    print(game_str)
    user_move = input('Enter move: ')
    if user_move == "invalid":
        game_str = game_str[:-len(bot_move)]
    else:
        game_str += user_move + ' '
    print(game_str)

    

    bot_move = ''
    while not bot_move.endswith(' '):
        pred = get_prediction(game_str + bot_move)
        bot_move += pred

    print('Bot plays: {}'.format(bot_move))
    game_str += bot_move
