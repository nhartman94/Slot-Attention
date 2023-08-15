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

class SlotAttentionEncoder(torch.nn.Module):
    def __init__(self, 
                 k_slots=3, 
                 hidden_dim=32, 
                 query_dim=32, 
                 slot_dim=32, 
                 n_iter=2,
                 device='cpu', 
                 final_mlp=False
                 ):
        '''
        Slot attention encoder block

        Inputs:
        - device (cpu, mps, cuda): Which device to put the model on 
                (needed for the random call when initializing the slots)
        - k_slots: number of slots (note, can vary between training and test time)
        ** To do:** Add a variable number of encoder blocks to apply!

        - hidden_dim: The hidden dimension for the CNN (currently single layer w/ no non-linearities)
        - query_dim: The latent space dimension that the slots and the queries get computed in
        - slot_dim:  
        - n_iter: Number of slot attention steps to apply (defualt 2, and rn hard coded)
        - final_mlp (bool, default False): Whether to add a final residual MLP after the GRUCell
        '''
        super().__init__()

        self.k_slots = k_slots
        self.hidden_dim = hidden_dim
        self.query_dim = query_dim
        self.slot_dim = slot_dim
        self.n_iter = n_iter

        self.device=device
        self.final_mlp = final_mlp

        self.queryN = torch.nn.LayerNorm(self.query_dim)
        self.dataN = torch.nn.LayerNorm(self.hidden_dim)
        self.updateN = torch.nn.LayerNorm(self.query_dim)

        self.toK = torch.nn.Linear(self.hidden_dim,self.query_dim)
        self.toV = torch.nn.Linear(self.hidden_dim,self.query_dim)
        self.gru = torch.nn.GRUCell(self.query_dim, self.query_dim)

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(5,self.hidden_dim,5, padding = 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.hidden_dim,self.hidden_dim,5, padding = 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.hidden_dim,self.hidden_dim,5, padding = 2),
        )   

        '''
        This is an mlp which gets applied to the final queries
        Q1: Where is this in the paper
        Q2: _Why_ is this necessary
        '''         
        # if self.final_mlp:
        #     self.final = torch.nn.Sequential(
        #         torch.nn.Linear(self.query_dim, 16),
        #         torch.nn.ReLU(),
        #         torch.nn.Linear(16,self.slot_dim)
        #     )
        # else:
        #     # Just apply the identity function
        #     self.final = lambda x : x

        # I think this is the one that is supposed to be optional in the
        self.updateff = torch.nn.Sequential(
            torch.nn.Linear(self.query_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,self.query_dim)
        )
        
    def init_slots(self, Nbatch):
        noise = torch.randn(Nbatch, self.k_slots, self.query_dim).to(self.device)
        
        mu = torch.zeros(1,1,self.query_dim).to(self.device)
        logsigma = torch.zeros(1,1,self.query_dim).to(self.device)
        
        return mu + noise*logsigma.exp()


    def attention_and_weights(self,queries,encoded_data):
        keys = self.toK(encoded_data)
        att = torch.einsum('bse,bde->bsd',queries,keys) * (self.query_dim ** (-0.5)) * 10
        att = torch.nn.functional.softmax(att, dim = 1)

        div = torch.sum(att, dim = -1, keepdims = True)
        wts = att/div + 1e-8
        return att,wts

    def iterate(self, queries, encoded_data):
        encoded_data = torch.permute(encoded_data,(0,2,3,1))
        encoded_data = torch.flatten(encoded_data,1,2)
        encoded_data = self.dataN(encoded_data)
        
        att,wts = self.attention_and_weights(self.queryN(queries),encoded_data)        
        vals = self.toV(encoded_data)
        extracted = torch.einsum('bsd,bde->bse',wts,vals)
        
        extracted = self.gru(
            extracted.reshape(-1,self.query_dim),
            queries.reshape(-1,self.query_dim),
        )

        extracted = extracted.reshape(queries.shape)
        return extracted + self.updateff(self.updateN(extracted))
    
    def forward(self, data):

        Nbatch = data.shape[0]
        queries = self.init_slots(Nbatch)
        encoded_data = self.encoder(data)

        for i in range(self.n_iter):
            queries = self.iterate(queries, encoded_data)  

        positions = self.final(queries) # final residual MLP

        return positions,queries
        
