import torch
import torch.nn as nn
import torch.nn.functional as F

from ..defaults import DEFAULT_DROPOUT, DEFAULT_EMBED, DEFAULT_BLOCK_SIZE
from .feedforward import FeedForward
from .selfattention import MultiHeadAttention

class Block(nn.Module):

    def __init__(self, n_heads, n_embed = DEFAULT_EMBED, block_size = DEFAULT_BLOCK_SIZE, dropout = DEFAULT_DROPOUT, masked = True):
        super().__init__()

        self.n_heads = n_heads
        self.head_size = n_embed // n_heads
        self.n_embed = n_embed
        self.block_size = block_size

        self.mha = MultiHeadAttention(n_heads, self.head_size, n_embed, block_size, dropout=dropout, masked=masked)
        self.ffwd = FeedForward(n_embed, dropout=dropout)

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x



class DecoderBlock(nn.Module):

    def __init__(self, n_heads, n_embed = DEFAULT_EMBED, block_size = DEFAULT_BLOCK_SIZE, dropout = DEFAULT_DROPOUT):
        super().__init__()

        self.n_heads = n_heads
        self.head_size = n_embed // n_heads
        self.n_embed = n_embed
        self.block_size = block_size

        self.mha = MultiHeadAttention(n_heads, self.head_size, n_embed, block_size, dropout=dropout,masked=True)
        self.mha2 = MultiHeadAttention(n_heads, self.head_size, n_embed, block_size, dropout=dropout, masked=False)
        self.ffwd = FeedForward(n_embed, dropout=dropout)

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ln3 = nn.LayerNorm(n_embed)

    def forward(self, x, encoder_output):
        x = x + self.mha(self.ln1(x))
        x = x + self.mha2(self.ln2(x+encoder_output))
        x = x + self.ffwd(self.ln3(x))

        return x