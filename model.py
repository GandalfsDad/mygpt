import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_EMBED = 32
DEFAULT_BLOCK_SIZE = 8
DEFAULT_HEADS = 4
DEFAULT_BLOCKS = 3
DEFAULT_DROPOUT = 0.2

class BigramLanguageModel(nn.Module):

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

class FeedForward(nn.Module):

    def __init__(self, n_embed, dropout=DEFAULT_DROPOUT):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_heads, n_embed = DEFAULT_EMBED, block_size = DEFAULT_BLOCK_SIZE, dropout = DEFAULT_DROPOUT):
        super().__init__()

        self.n_heads = n_heads
        self.head_size = n_embed // n_heads
        self.n_embed = n_embed
        self.block_size = block_size

        self.mha = MultiHeadAttention(n_heads, self.head_size, n_embed, block_size, dropout=dropout)
        self.ffwd = FeedForward(n_embed, dropout=dropout)

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x