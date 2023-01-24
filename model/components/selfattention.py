import torch
import torch.nn as nn
import torch.nn.functional as F

from ..defaults import DEFAULT_DROPOUT, DEFAULT_EMBED, DEFAULT_BLOCK_SIZE


class Head(nn.Module):

    def __init__(self, head_size, n_embed = DEFAULT_EMBED, block_size = DEFAULT_BLOCK_SIZE, dropout = DEFAULT_DROPOUT):
        super().__init__()

        self.head_size = head_size
        self.n_embed = n_embed
        self.block_size = block_size

        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                                             #(B, T, C)
        q = self.query(x)                                           #(B, T, C)

        wei = k @ q.transpose(-2,-1) / (C ** 0.5)                   #(B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) #(B, T, T)
        wei = F.softmax(wei, dim=-1)                                #(B, T, T) 
        wei = self.dropout(wei)                                     #(B, T, T)

        v = self.value(x)                                           #(B, T, C)
        out = wei @ v                                               #(B, T, C)

        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, head_size, n_embed = DEFAULT_EMBED, block_size = DEFAULT_BLOCK_SIZE, dropout = DEFAULT_DROPOUT):
        super().__init__()

        self.n_heads = n_heads
        self.head_size = head_size
        self.n_embed = n_embed
        self.block_size = block_size

        self.heads = nn.ModuleList([Head(head_size, n_embed, block_size, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads] , dim=-1)
        out = self.dropout(self.proj(out))

        return out