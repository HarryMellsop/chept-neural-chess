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
block_size = 512

# save the device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)

# load pretrain dataset
games = open('data/datasets-cleaned/kingbase_cleaned.txt').read()
#games = games[:int(len(games) / 10)]
pretrain_dataset = dataset.PretrainDataset(games, block_size=block_size)

# load model
mconf = model.GPTConfig(
    vocab_size=pretrain_dataset.vocab_size, 
    block_size=pretrain_dataset.block_size, 
    n_layer=12, 
    n_head=16,
    n_embd=256
)
model = model.GPT(mconf)

train_config = trainer.TrainerConfig(
    max_epochs=1,
    batch_size=64,
    learning_rate=1e-3,
    num_workers=4
)

trainer = trainer.Trainer(model, pretrain_dataset, train_config)
trainer.train()
