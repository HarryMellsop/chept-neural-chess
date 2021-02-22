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
model.load_state_dict(torch.load('ckpt/pretrain.model.params', map_location=torch.device('cpu')))

# load dataset
with open('cache/stoi.pkl', 'rb') as f: 
    stoi = pickle.load(f)
with open('cache/itos.pkl', 'rb') as f:
    itos = pickle.load(f)

cur_game = ''
PAD_CHAR = u'\u25A1'

# run inference loop
while True:
    print('Welcome to Chess Bot. Enter moves below to start a game.')

    user_submission = input('Enter move: ')
    if user_submission is 'quit': break

    print(itos)
    game_str = '1.c4 c5 2.Nc3 Nc6'

    x = game_str + PAD_CHAR * (block_size - len(game_str))
    x = torch.tensor([stoi[c] for c in x], dtype=torch.long)
    x = x.view(1, -1)

    model.eval()
    with torch.no_grad():
        logits, _ = model(x)
        logits = torch.squeeze(logits)
        y_hat = torch.argmax(logits, dim=-1)
        y_hat = [itos[y_hat[t].item()] for t in y_hat]
        print(''.join(y_hat))