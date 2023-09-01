'''
model.py

Code from Lukas Heinrich 
(functions from the pytorch.py nobotebook galvanized from the )

'''
import numpy as np
import torch
from torch.nn import init

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
    def __init__(self,hidden_dim,resolution,device='cpu',pixel_mult=1):
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
        self.pixel_mult = pixel_mult
        grid = build_grid(resolution)
        self.grid = torch.FloatTensor( grid ).to(device)
        

    def forward(self, x):
        
        return x + self.pixel_mult * self.dense(self.grid)
       
class SlotAttentionPosEmbed(torch.nn.Module):
    def __init__(self, 
                 resolution=(32,32),
                 k_slots=3, 
                 num_conv_layers=3,
                 hidden_dim=32, 
                 final_cnn_relu=False,
                 learn_init=False,
                 query_dim=32, 
                 n_iter=2,
                 softmax_T='defaultx10',
                 pixel_mult=1,
                 pos_inpts=False,
                 learn_slot_feat=False,
                 device='cpu',
                 ):
        '''
        Slot attention encoder block with positional embedding

        Inputs:
        - device (cpu (default), mps, cuda): Which device to put the model on 
                (needed for the random call when initializing the slots)
        - k_slots (default 3): number of slots (note, can vary between training and test time)
        - num_conv_layers: # of convolutional layers to apply (google paper has 4)
        - hidden_dim (default 32): The hidden dimension for the CNN (currently single layer w/ no non-linearities)
        - final_cnn_relu: Whether to apply the final cnn relu for these experiments (use true to mimic google repo)
        - query_dim (default 32): The latent space dimension that the slots and the queries get computed in
        - n_iter (default  2): Number of slot attention steps to apply (defualt 2)
        - T (str): Softmax temperature for scaling the logits 
            Needs to be one of 3 options:
            * default: 1/sqrt(query_dim)
            * defaultx10: 1/sqrt(query_dim)*10 (from LH WTFAE nb)
            * 1/D: 1/query_dim, as suggested by muTransfer 2203.03466
        - device (str): Which device to put the model on.
            Also used when drawing random samples for the query points 
            and the grid generation for the positional encoding
        '''
        super().__init__()

        
        self.k_slots = k_slots
        self.hidden_dim = hidden_dim
        self.query_dim = query_dim
        self.n_iter = n_iter

        self.resolution = resolution

        self.device=device
        
        assert softmax_T in ['default','defaultx10','1/D'] 
        if softmax_T=='default':
            self.softmax_T = 1/np.sqrt(query_dim)
        elif softmax_T=='defaultx10':
            self.softmax_T = 10/np.sqrt(query_dim)
        elif softmax_T=='1/D':
            self.softmax_T = 1/query_dim
        else:
            print(f'Softmax temperature {T} not supported')
            raise SyntaxError
        
        self.dataN = torch.nn.LayerNorm(self.hidden_dim)
        self.queryN = torch.nn.LayerNorm(self.query_dim)
        
        self.toK = torch.nn.Linear(self.hidden_dim, self.query_dim)
        self.toV = torch.nn.Linear(self.hidden_dim, self.query_dim)
        self.gru = torch.nn.GRUCell(self.query_dim, self.query_dim)

        input_dim = 5 if pos_inpts else 1 # whether or not to include the pixels w/ the input
        kwargs = { 'out_channels': hidden_dim,'kernel_size': 5, 'padding':2 }
        cnn_layers = [torch.nn.Conv2d(input_dim,**kwargs)]
        for i in range(num_conv_layers-1):
            cnn_layers += [torch.nn.ReLU(), torch.nn.Conv2d(hidden_dim,**kwargs)] 
            
        if final_cnn_relu:
            '''
            22.08.2023 Silly mistake, I didn't include a ReLU() after the last CNN filter 
            for first exps. This `final_cnn_relu` flag is a hack for bkwds compatibility.
            '''
            cnn_layers.append(torch.nn.ReLU())

        self.CNN_encoder = torch.nn.Sequential(*cnn_layers)
          
        self.posEnc = SoftPositionalEmbed(hidden_dim, resolution,device, pixel_mult)
        
        if pos_inpts:
            self.process_data = lambda X: \
                torch.cat([X,
                torch.tile(self.posEnc.grid.permute(0,3,1,2), [X.shape[0],1,1,1])],dim=1)
        else:
            self.process_data = lambda X: X

        self.init_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim,hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim,hidden_dim)
        )

        if learn_init:
            self.slots_mu = torch.nn.Parameter(torch.randn(1, 1, self.query_dim))
            self.slots_logsigma = torch.nn.Parameter(torch.zeros(1, 1, self.query_dim))
            init.xavier_uniform_(self.slots_logsigma)

            self.init_slots = self.SA_init_slots

        else:
            self.init_slots = self.LH_init_slots

        '''
        Option to add a final (x,y,r) prediction to each slot
        '''
        self.learn_slot_feat = learn_slot_feat
        if self.learn_slot_feat:
            self.final_mlp = torch.nn.Sequential(
                torch.nn.Linear(query_dim,hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 3)
            )


    def SA_init_slots(self,Nbatch):
        '''
        Slot init taken from
        https://github.com/lucidrains/slot-attention/blob/master/slot_attention/slot_attention.py
        '''
        mu = self.slots_mu.expand(Nbatch, self.k_slots, -1)
        sigma = self.slots_logsigma.exp().expand(Nbatch, self.k_slots, -1)

        return mu + sigma * torch.randn(mu.shape).to(self.device)

    def LH_init_slots(self, Nbatch):
        noise = torch.randn(Nbatch, self.k_slots, self.query_dim).to(self.device)
        
        mu = torch.zeros(1,1,self.query_dim).to(self.device)
        logsigma = torch.zeros(1,1,self.query_dim).to(self.device)
        
        return mu + noise*logsigma.exp()
    
    def encoder(self,data):
        
        # If pos_inpts was passed at initialization, concatenate the grid
        encoded_data = self.process_data(data)

        # Apply the CNN encoder
        encoded_data = self.CNN_encoder(data)
        
        # Put the channel dim at the end
        encoded_data = torch.permute(encoded_data,(0,2,3,1)) 
                 
        # Add the positional embeddings
        encoded_data = self.posEnc(encoded_data)
        
        # Flatten the pixel dims and apply the data normalization + MLP
        encoded_data = torch.flatten(encoded_data,1,2)
        encoded_data = self.dataN(encoded_data)
        encoded_data = self.init_mlp(encoded_data)
        
        return encoded_data
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients.append( grad )
    
    def attention_and_weights(self,queries,encoded_data):
        keys = self.toK(encoded_data)
        logits = torch.einsum('bse,bde->bsd',queries,keys) * self.softmax_T
        
        att = torch.nn.functional.softmax(logits, dim = 1)

        if att.requires_grad:
            h = logits.register_hook(self.activations_hook)
        
        div = torch.sum(att, dim = -1, keepdims = True)
        wts = att/div + 1e-8
        return att,wts

    def iterate(self, queries, encoded_data):
        
        # queries: (bs, k_slots, query_dim)
        
        att,wts = self.attention_and_weights(self.queryN(queries),encoded_data)   
        
        # att: (bs, k_slots, img_dim)
        # wts: (bs, k_slots, img_dim)
        
        vals = self.toV(encoded_data) # bs, img_dim, query_dim
        updates = torch.einsum('bsd,bde->bse',wts,vals) # bs, n_slots, query_dim
        
        updates = self.gru(
            updates.reshape(-1,self.query_dim),
            queries.reshape(-1,self.query_dim),
        )

        return updates.reshape(queries.shape)
        
    def forward(self, data):

        self.gradients = []
    
        # Initialize the queries
        Nbatch = data.shape[0]
        queries = self.init_slots(Nbatch) # Shape (Nbatch, k_slots, query_dim)
        
        encoded_data = self.encoder(data)
        
        for i in range(self.n_iter):
            queries = self.iterate(queries, encoded_data)    
            
        # Then with the _final_ query vector, calc what the attn + weights would be
        att, wts = self.attention_and_weights(self.queryN(queries),encoded_data)   
            
        if self.learn_slot_feat:
            slot_feat = self.final_mlp(queries)
            return queries, att, slot_feat 
        
        else:
            return queries, att, wts