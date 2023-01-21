import torch

class Batcher:

    def __init__(self, train, val, block_size):
        self.train = train
        self.val = val
        self.block_size = block_size

    def get_batch(self, batch_size, train=True, device='cpu'):
        if train:
            data = self.train
        else:
            data = self.val

        idx = torch.randint(0,len(data)-self.block_size,(batch_size,))

        x = torch.stack([data[i:i+self.block_size] for i in idx])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in idx])

        x = x.to(device)
        y = y.to(device)
        return x, y 

        