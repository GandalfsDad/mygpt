from util import load_dataset, get_encode_decode, train_val_split
from batch import Batcher
from model import BigramLanguageModel

import torch
import torch.nn.functional as F
import torch.nn as nn

torch.manual_seed(69420)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

block_size = 128        #256
batch_size = 64         #128
learning_rate = 3e-4    #3e-4
max_len = 1000          #1000
train_frac = 0.9        #0.9
training_steps = 5000   #5000
eval_interval = 10      #100
eval_iters = 20         #100
n_embed = 128           #384
n_head = 4              #6
n_blocks = 4            #6
dropout = 0.2           #0.2

text = load_dataset()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}\nVocab {''.join(chars)}")

encode, decode = get_encode_decode(chars)
data = torch.tensor(encode(text), dtype=torch.long)

train_data, val_data = train_val_split(data, train_frac)

batcher = Batcher(train_data, val_data, block_size)
model = BigramLanguageModel(vocab_size, n_embed = n_embed, block_size = block_size, n_heads = n_head, n_blocks = n_blocks, dropout = dropout)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

model = model.to(device)

@torch.no_grad()
def estimate_loss():
   out = {}
   model.eval()
   for split in ['train','val']:
      losses = torch.zeros(eval_iters)
      for k in range(eval_iters):
         x,y = batcher.get_batch(batch_size, train=split=='train', device=device)
         logits, loss = model(x,y)
         losses[k] = loss.item()
      out[split] = losses.mean()
   model.train()
   return out

         
for step in range(training_steps):

    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step: {step} Train Loss: {losses['train']} Val Loss: {losses['val']}")

    x,y = batcher.get_batch(batch_size, train=True, device = device)
    logits, loss = model(x,y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    
print(f"Final Loss {loss.item()}")


print(f"Generating Sample")
idx = torch.zeros((1,1), dtype=torch.long, device= device)
decode(model.generate(idx,max_len=max_len)[0].tolist())
print(decode)