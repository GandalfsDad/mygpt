import torch
import torch.nn as nn
import torch.nn.functional as F

from .defaults import DEFAULT_DROPOUT, DEFAULT_EMBED, DEFAULT_BLOCK_SIZE, DEFAULT_HEADS, DEFAULT_BLOCKS
from .components.block import Block

class Translator(nn.Module):

    def __init__(self):
        pass

    def forward(self, idx, targets = None):
        pass