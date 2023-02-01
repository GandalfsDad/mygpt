import torch
import torch.nn as nn
import torch.nn.functional as F

from ..defaults import DEFAULT_DROPOUT, DEFAULT_EMBED, DEFAULT_BLOCK_SIZE, DEFAULT_HEADS, DEFAULT_BLOCKS
from .block import Block

class Encoder(nn.Module):

    def __init__(self, vocab_size, n_embed = DEFAULT_EMBED, block_size = DEFAULT_BLOCK_SIZE, n_heads = DEFAULT_HEADS, n_blocks = DEFAULT_BLOCKS, dropout = DEFAULT_DROPOUT):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.dropout = dropout
        
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)

        self.blocks = nn.Sequential(
            *[Block(n_heads,n_embed, block_size, dropout, masked=False) for _ in range(n_blocks)],
            nn.LayerNorm(n_embed)
        )

    def forward(self, idx):
        B, T = idx.shape

        token_embed = self.token_embedding(idx) #(B, T, C)
        pos_embed = self.position_embedding(torch.arange(T)) #(T, C)

        x = token_embed + pos_embed #(B, T, C)
        x = self.blocks(x) #(B, T, C)

        return x