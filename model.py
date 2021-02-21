import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class GPTConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    additive = False

    def __init__(self, vocab_size, block_size, attention_type, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.attention_type = attention_type
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    n_layer = 12
    n_head = 12
    n_embd = 768


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.attn_pdrop,
            bias=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.block_size = config.block_size

        print("Number of parameters: {}".format(sum(p.numel() for p in self.parameters())))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx):
        b, t = idx.size()

        # forward the GPT model
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        return logits