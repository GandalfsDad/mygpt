import torch
import torch.nn as nn
import torch.nn.functional as F

from .defaults import DEFAULT_DROPOUT, DEFAULT_EMBED, DEFAULT_BLOCK_SIZE, DEFAULT_HEADS, DEFAULT_BLOCKS
from .components.encoder import Encoder
from .components.decoder import Decoder

class Translator(nn.Module):

    def __init__(self, encoder_vocab_size, decoder_vocab_size, n_embed = DEFAULT_EMBED, block_size = DEFAULT_BLOCK_SIZE, n_heads = DEFAULT_HEADS, n_blocks = DEFAULT_BLOCKS, dropout = DEFAULT_DROPOUT):
        super().__init__()

        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.dropout = dropout
        
        self.encoder_token_embedding = nn.Embedding(encoder_vocab_size, n_embed)
        self.encoder_position_embedding = nn.Embedding(block_size, n_embed)

        self.decoder_token_embedding = nn.Embedding(decoder_vocab_size, n_embed)
        self.decoder_position_embedding = nn.Embedding(block_size, n_embed)

        self.encoder = Encoder(n_embed, block_size, n_heads, n_blocks, dropout)
        self.decoder = Decoder(n_embed, block_size, n_heads, n_blocks, dropout)


    def forward(self, idx, targets):
        
        encoder_output = self.encoder(idx)
        decoder_output = self.decoder(targets, encoder_output)

        B, T, C = decoder_output.shape
        decoder_output = decoder_output.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(decoder_output, targets)

        return decoder_output, loss