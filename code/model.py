'''
model.py

Code from Lukas Heinrich 
(functions from the pytorch.py nobotebook galvanized from the )

'''
import numpy as np
import torch

def build_grid(resolution):
    '''
    From google slot attention repo:
    https://github.com/nhartman94/google-research/blob/master/slot_attention/model.py#L357C1-L364C53
    '''
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return np.concatenate([grid, 1.0 - grid], axis=-1)


class SoftPositionalEmbed(torch.nn.Module):
    def __init__(self,hidden_dim,resolution,device='cpu'):
        '''
        Given the dimensions of the input image, this layer adds 
        a residual connection for a _learnable projection_ of the
        grid of the (normalized) positions of the input image.
        
        - hidden_dim: The # of channels that the input image has
        - resolution: A tuple of the width and height of the input image.
        
        Translated into pytorch from google's tf fct:
        https://github.com/nhartman94/google-research/blob/master/slot_attention/model.py#L367-L382
        
        '''
        super().__init__()
        
        self.dense = torch.nn.Linear(4, hidden_dim)
        
        grid = build_grid(resolution)
        self.grid = torch.FloatTensor( grid ).to(device)
        
    def forward(self, x):
        
        return x + self.dense(self.grid)


       
