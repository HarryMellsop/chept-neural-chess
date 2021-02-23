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
vocab_size = 40
block_size = 1024

# save the device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# build model config
mconf = model.GPTConfig(
    vocab_size=vocab_size, 
    block_size=block_size, 
    n_layer=4, 
    n_head=8, 
    n_embd=256
)

# load model weights
model = model.GPT(mconf)
# model.load_state_dict(torch.load('ckpt/pretrain.model.params', map_location=torch.device('cpu')))

# load dataset
with open('cache/stoi.pkl', 'rb') as f: 
    stoi = pickle.load(f)
with open('cache/itos.pkl', 'rb') as f:
    itos = pickle.load(f)

cur_game = ''
PAD_CHAR = u'\u25A1'

def get_prediction(game_str):

    x = game_str + PAD_CHAR * (block_size - len(game_str))
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
move_it = 1
print('Welcome to Chess Bot. Enter moves below to start a game.')

while True:
    user_move = input('Enter move: ')

    game_str += str(move_it) + '.'
    game_str += user_move + ' '

    bot_move = ''
    while not bot_move.endswith(' '):
        pred = get_prediction(game_str + bot_move)
        bot_move += pred

    print('Bot plays: {}'.format(bot_move))
    game_str += bot_move
