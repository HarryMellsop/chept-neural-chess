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

# LOAD HYPERPARAMS
block_size = 1024

# save the device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# load pretrain dataset
games = open('data/datasets-cleaned/kingbase_cleaned.txt').read()
games = games[:int(len(games) / 10)]
pretrain_dataset = dataset.ChessMoveDataset(games, block_size=block_size)

# load model
mconf = model.GPTConfig(
    vocab_size=pretrain_dataset.vocab_size, 
    block_size=pretrain_dataset.block_size, 
    n_layer=4, 
    n_head=8, 
    n_embd=256
)
model = model.GPT(mconf)

train_config = trainer.TrainerConfig(
    max_epochs=5,
    batch_size=16,
    learning_rate=6e-3,
    lr_decay=True, 
    warmup_tokens=512*20, 
    final_tokens=200 * len(pretrain_dataset) * block_size,
    num_workers=4
)

trainer = trainer.Trainer(model, pretrain_dataset, None, train_config)
trainer.train()
