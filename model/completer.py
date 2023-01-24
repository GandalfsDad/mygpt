import torch
import torch.nn as nn
import torch.nn.functional as F

from .defaults import DEFAULT_DROPOUT, DEFAULT_EMBED, DEFAULT_BLOCK_SIZE, DEFAULT_HEADS, DEFAULT_BLOCKS
from .components.block import Block

class Completer(nn.Module):

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
            *[Block(n_heads,n_embed, block_size, dropout) for _ in range(n_blocks)],
            nn.LayerNorm(n_embed)
        )

        self.lm_head = nn.Linear(n_embed, vocab_size)


    def forward(self, idx, targets = None):
        B, T = idx.shape

        token_embed = self.token_embedding(idx) #(B, T, C)
        pos_embed = self.position_embedding(torch.arange(T)) #(T, C)

        x = token_embed + pos_embed #(B, T, C)
        x = self.blocks(x) #(B, T, C)

        logits = self.lm_head(x) #(B, T, vocab_size)

        if targets is None:
            loss = None 
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_len=100):
        for _ in range(max_len):

            idx_subset = idx[:,-self.block_size:]
            logits, loss = self(idx_subset)
            logits = logits[:,-1,:] #last timestep only
            probs = F.softmax(logits, dim= -1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
