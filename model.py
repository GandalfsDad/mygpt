import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embed = 32, block_size = 8):
        super().__init__()
        self.vocab_size = vocab_size
        
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape

        token_embed = self.token_embedding(idx) #(B, T, C)
        pos_embed = self.position_embedding(torch.arange(T)) #(T, C)

        x = token_embed + pos_embed #(B, T, C)
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
            logits, loss = self(idx)
            logits = logits[:,-1,:] #last timestep only
            probs = F.softmax(logits, dim= -1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    