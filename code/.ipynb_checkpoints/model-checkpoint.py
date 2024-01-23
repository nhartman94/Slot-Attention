'''
model.py

Code from Lukas Heinrich 
(functions from the pytorch.py nobotebook galvanized from the )

'''
import numpy as np
import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F

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

        # The parameters for the slot initialization
        self.slots_mu = torch.nn.Parameter(torch.randn(1, 1, self.query_dim))
        self.slots_logsigma = torch.nn.Parameter(torch.zeros(1, 1, self.query_dim))
        init.xavier_uniform_(self.slots_logsigma)

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


    def init_slots(self,Nbatch):
        '''
        Slot init taken from
        https://github.com/lucidrains/slot-attention/blob/master/slot_attention/slot_attention.py
        '''
        mu = self.slots_mu.expand(Nbatch, self.k_slots, -1)
        sigma = self.slots_logsigma.exp().expand(Nbatch, self.k_slots, -1)

        return mu + sigma * torch.randn(mu.shape).to(self.device)

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
    
    
class BasicBlock(nn.Module):
    """Basic Residual Block"""
    def __init__(self, inplanes, outplanes, stride=1, kernel_size=3, padding=0):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        
        self.sampling = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.stride = stride
        
        #self.linear output: 16*32*32 -> reshape it afterwards into [bs, 16, 32, 32]
        
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        #print("identity: ", x.shape) # [bs, 8, 32, 32]
        #print("out: ", out.shape)  # [bs, 16, 32, 32]
        if(identity.shape!=out.shape):
            identity = self.sampling(x) # is that ok? -> Nicole! 
            
        out += identity # this is the trick!!
        out = self.relu(out)

        return out
    


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 8,kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        # I choose my conv layers in a way that the final output will be [32, 32] still
        
        # FIRST STAGE -  capture basic features and patterns in the input while reducing its spatial resolution (second not needed?)
        self.layer1 = BasicBlock(8, 8,kernel_size=5, stride=1, padding=2)
        # SECOND STAGE - capture more complex features and patterns compared to the initial stage. The spatial dimensions are reduced, but the number of channels (depth) is increased.
        self.layer2 = BasicBlock(8, 16, kernel_size=3, stride=1, padding=1)
        # THIRD STAGE -  capture more abstract features and high-level representations, as the spatial dimensions continue to decrease.
        self.layer3 = BasicBlock(16, 32, kernel_size=3, stride=1, padding=1)
        # FOUTH STAGE- - capture very abstract and global features, consolidating the information learned from the previous stages.
        self.layer4 = BasicBlock(32, 16, kernel_size=3, stride=1, padding=1)
        
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = self.avgpool(x)
        #print("shape after resnet block: ", x.shape)
        return x
    
class BigResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 8,kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        # I choose my conv layers in a way that the final output will be [32, 32] still
        
        # FIRST STAGE -  capture basic features and patterns in the input while reducing its spatial resolution (second not needed?)
        self.layer11 = BasicBlock(8, 8,kernel_size=5, stride=1, padding=2)
        self.layer12 = BasicBlock(8, 8,kernel_size=5, stride=1, padding=2)
        self.layer13 = BasicBlock(8, 8,kernel_size=5, stride=1, padding=2)
        self.layer14 = BasicBlock(8, 8,kernel_size=5, stride=1, padding=2)
        # SECOND STAGE - capture more complex features and patterns compared to the initial stage. The spatial dimensions are reduced, but the number of channels (depth) is increased.
        self.layer21 = BasicBlock(8, 16, kernel_size=3, stride=1, padding=1)
        self.layer22 = BasicBlock(16, 16, kernel_size=3, stride=1, padding=1)
        self.layer23 = BasicBlock(16, 16, kernel_size=3, stride=1, padding=1)
        self.layer24 = BasicBlock(16, 16, kernel_size=3, stride=1, padding=1)
        # THIRD STAGE -  capture more abstract features and high-level representations, as the spatial dimensions continue to decrease.
        self.layer31 = BasicBlock(16, 32, kernel_size=3, stride=1, padding=1)
        self.layer32 = BasicBlock(32, 32, kernel_size=3, stride=1, padding=1)
        self.layer33 = BasicBlock(32, 32, kernel_size=3, stride=1, padding=1)
        self.layer34 = BasicBlock(32, 32, kernel_size=3, stride=1, padding=1)
        # FOUTH STAGE- - capture very abstract and global features, consolidating the information learned from the previous stages.
        self.layer4 = BasicBlock(32, 16, kernel_size=3, stride=1, padding=1)
        
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer21(x)
        x = self.layer22(x)
        x = self.layer23(x)
        x = self.layer24(x)
        x = self.layer31(x)
        x = self.layer32(x)
        x = self.layer33(x)
        x = self.layer34(x)
        x = self.layer4(x)

        #x = self.avgpool(x)
        #print("shape after resnet block: ", x.shape)
        return x





class InvariantSlotAttention(torch.nn.Module):
    def __init__(self, 
                 resolution=(32,32),
                 xlow=-0.5,
                 xhigh=0.5,
                 varlow=0.01,
                 varhigh=0.05,
                 k_slots=3, 
                 num_conv_layers=3,
                 which_encoder='',
                 hidden_dim=32, 
                 query_dim=32, 
                 n_iter=2,
                 pixel_mult=1,
                 device='cpu' ,
                 learn_slot_feat=True
                 ):
        '''
        Slot attention encoder block, block attention
        '''
        super().__init__()

        self.k_slots = k_slots
        self.hidden_dim = hidden_dim
        self.query_dim = query_dim
        self.n_iter = n_iter

        self.resolution = resolution
        self.xlow, self.xhigh = xlow, xhigh
        self.rlow, self.rhigh = np.sqrt(varlow), np.sqrt(varhigh)
        
        self.device=device
         
        self.softmax_T = 1/np.sqrt(query_dim)
        
        self.dataN = torch.nn.LayerNorm(self.hidden_dim)
        self.queryN = torch.nn.LayerNorm(self.query_dim)
        
        self.toK = torch.nn.Linear(self.hidden_dim, self.query_dim)
        self.toV = torch.nn.Linear(self.hidden_dim, self.query_dim)
        self.gru = torch.nn.GRUCell(self.query_dim, self.query_dim)
        
        '''
        CNN feature extractor
        '''
        kwargs = {'out_channels': hidden_dim,'kernel_size': 5, 'padding':2 }
        cnn_layers = [torch.nn.Conv2d(1,**kwargs)]
        for i in range(num_conv_layers-1):
            cnn_layers += [torch.nn.ReLU(), torch.nn.Conv2d(hidden_dim,**kwargs)] 
        cnn_layers.append(torch.nn.ReLU())
          
        self.CNN_encoder = torch.nn.Sequential(*cnn_layers) # 3 CNN layers by default
        if which_encoder=='MyResNet':
            self.CNN_encoder = ResNet()
        elif which_encoder=='MyBigResNet':
            self.CNN_encoder = BigResNet()
        elif which_encoder=='ResNet_Sanz1':
            self.CNN_encoder = Encoder_resnet_S1()
        elif which_encoder=='ResNet_Sanz2':
            self.CNN_encoder = Encoder_resnet_S2()
        print("Using " + which_encoder + " to encode data.")
            
          
            
        # Grid + query init
        self.abs_grid = self.build_grid()
                   
        self.dense = torch.nn.Linear(2, query_dim) 
        self.pixel_mult = pixel_mult # LH's proposal... but almost same as 1/delta in ISA

        # Apply after the data normalization
        self.init_mlp = torch.nn.Sequential(
            torch.nn.Linear(query_dim,query_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(query_dim,query_dim)
        )
            
        self.slots_mu = torch.nn.Parameter(torch.randn(1, 1, self.query_dim))
        self.slots_logsigma = torch.nn.Parameter(torch.zeros(1, 1, self.query_dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.init_slots = self.init_slots

        
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
        
    def build_grid(self):
        '''
        From google slot attention repo:
        https://github.com/nhartman94/google-research/blob/master/slot_attention/model.py#L357C1-L364C53
        '''
        resolution = self.resolution
        xlow, xhigh = self.xlow, self.xhigh
           
        ranges = [np.linspace(xlow, xhigh, num=res) for res in resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="xy")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [resolution[0], resolution[1], -1])
        grid = np.expand_dims(grid, axis=0)
        
        grid = torch.FloatTensor( grid ).to(self.device)
        grid = torch.flatten(grid,1,2)
    
        return grid
                
    def init_slots(self,Nbatch):
        '''
        Slot init taken from
        https://github.com/lucidrains/slot-attention/blob/master/slot_attention/slot_attention.py
        '''
        
        stdhigh, stdlow = self.rlow, self.rhigh
        
        mu = self.slots_mu.expand(Nbatch, self.k_slots, -1)
        sigma = self.slots_logsigma.exp().expand(Nbatch, self.k_slots, -1)
    
        queries = mu + sigma * torch.randn(mu.shape,device=self.device)
    
        # Add the position and scale initialization for the local ref frame
        ref_frame_dim = 3
        pos_scale = torch.rand(Nbatch, self.k_slots, ref_frame_dim,device=self.device)

        pos_scale[:,:2] -= 0.5
        pos_scale[:,-1]  = (stdhigh - stdlow) * pos_scale[:,-1] + stdlow
        
        return queries, pos_scale
     
    def get_keys_vals(self, encoded_data, pos_scale):

        # Get the relative position embedding
        rel_grid = self.abs_grid.unsqueeze(1) - pos_scale[:,:,:2].unsqueeze(2)
        rel_grid /= pos_scale[:,:,-1].unsqueeze(2).unsqueeze(-1)
        
        # Embed it in the same space as the query dimension 
        embed_grid = self.pixel_mult * self.dense( rel_grid )
        
        # keys, vals: (bs, img_dim, query_dim)
        keys = self.toK(encoded_data).unsqueeze(1) + embed_grid
        vals = self.toV(encoded_data).unsqueeze(1) + embed_grid
        
        keys = self.init_mlp(self.queryN(keys))
        vals = self.init_mlp(self.queryN(vals))
        
        return keys, vals
                
    def attention_and_weights(self,queries,keys):
        
        logits = torch.einsum('bse,bsde->bsd',queries,keys) * self.softmax_T
        
        att = torch.nn.functional.softmax(logits, dim = 1)
        
        div = torch.sum(att, dim = -1, keepdims = True)
        wts = att/div + 1e-8
        return att,wts

    def update_frames(self,wts):
        '''
        Update the relative frame position
        '''
        
        # expand to include the batch dim
        grid_exp = self.abs_grid.expand(wts.shape[0],-1,2)
        
        new_pos = torch.einsum('bsd,bde->bse',wts,grid_exp)
        
        new_scale = torch.sum(torch.pow(grid_exp.unsqueeze(1) - new_pos.unsqueeze(2),2),dim=-1)
        
        new_scale = torch.einsum('bsd,bsd->bs', wts, new_scale)
        new_scale = torch.sqrt(new_scale)
        
        return torch.cat([new_pos,new_scale.unsqueeze(-1)],axis=-1)
        
    def iterate(self, queries, pos_scale, encoded_data):
        
        # Get the keys and values in the ref ref frame
        keys, vals = self.get_keys_vals(encoded_data,pos_scale)
        
        # att,wts: (bs, k_slots, img_dim)
        att,wts = self.attention_and_weights(self.queryN(queries),keys)   
        
        new_pos_scale = self.update_frames(wts)
        
        # Update the queries with the recurrent block
        updates = torch.einsum('bsd,bsde->bse',wts,vals) # bs, n_slots, query_dim
        
        updates = self.gru(
            updates.reshape(-1,self.query_dim),
            queries.reshape(-1,self.query_dim),
        )
        
        return updates.reshape(queries.shape), new_pos_scale
        
    def forward(self, data):
    
        '''
        Step 1: Extract the CNN features
        '''
        encoded_data = self.CNN_encoder(data) # Apply the CNN encoder
        encoded_data = torch.permute(encoded_data,(0,2,3,1)) # Put channel dim at the end
        encoded_data = torch.flatten(encoded_data,1,2) # flatten pixel dims
        encoded_data = self.dataN(encoded_data)
        
        '''
        Step 2: Initialize the slots
        '''
        Nbatch = data.shape[0]
        queries, pos_scale = self.init_slots(Nbatch) # Shape (Nbatch, k_slots, query_dim)
                
        '''
        Step 3: Iterate through the reconstruction
        '''
        for i in range(self.n_iter):
            queries, pos_scale = self.iterate(queries, pos_scale, encoded_data)    
            
        # With the final query vector, calc the attn, weights, + rel ref frames
        keys, vals = self.get_keys_vals(encoded_data,pos_scale)
        att, wts = self.attention_and_weights(self.queryN(queries),keys)   
        new_pos_scale = self.update_frames(wts)
                
        if self.learn_slot_feat:
            slot_feat = self.final_mlp(queries)
            
            # Want to learn the delta from the previously estimated position
            slot_feat += new_pos_scale
            
            return queries, att, slot_feat 
        
        else:
            return queries, att, wts

class InvariantSlotAttention_disc(torch.nn.Module):
    def __init__(self, 
                 resolution=(32,32),
                 xlow=-0.5,
                 xhigh=0.5,
                 varlow=0.01,
                 varhigh=0.05,
                 k_slots=3, 
                 num_conv_layers=3,
                 alpha_depth = 32,
                 hidden_dim=32, 
                 query_dim=32, 
                 n_iter=2,
                 pixel_mult=1,
                 device='cpu' ,
                 learn_slot_feat=True
                 ):
        '''
        Slot attention encoder block, block attention
        '''
        super().__init__()

        self.k_slots = k_slots
        self.hidden_dim = hidden_dim # dimension after CNN encoder 
        self.query_dim = query_dim # dimension in which slot attention lives
        self.n_iter = n_iter

        self.resolution = resolution
        self.xlow, self.xhigh = xlow, xhigh
        self.rlow, self.rhigh = np.sqrt(varlow), np.sqrt(varhigh)
        
        self.device=device
         
        self.softmax_T = 1/np.sqrt(query_dim)
        
        self.dataN = torch.nn.LayerNorm(self.hidden_dim)
        self.queryN = torch.nn.LayerNorm(self.query_dim)
        
        self.toK = torch.nn.Linear(self.hidden_dim, self.query_dim)
        self.toV = torch.nn.Linear(self.hidden_dim, self.query_dim)
        self.gru = torch.nn.GRUCell(self.query_dim, self.query_dim)
        
        '''
        CNN feature extractor
        '''
        kwargs = {'out_channels': hidden_dim,'kernel_size': 5, 'padding':2 }
        cnn_layers = [torch.nn.Conv2d(1,**kwargs)]
        for i in range(num_conv_layers-1):
            cnn_layers += [torch.nn.ReLU(), torch.nn.Conv2d(hidden_dim,**kwargs)] 
        cnn_layers.append(torch.nn.ReLU())

        self.CNN_encoder = torch.nn.Sequential(*cnn_layers) # this is CNN model! 
            
        # Grid + query init
        self.abs_grid = self.build_grid()
                   
        self.dense = torch.nn.Linear(2, query_dim) 
        self.pixel_mult = pixel_mult # LH's proposal... but almost same as 1/delta in ISA

        # Apply after the data normalization
        self.init_mlp = torch.nn.Sequential( # what's this for???
            torch.nn.Linear(query_dim,query_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(query_dim,query_dim)
        )
            
        self.slots_mu = torch.nn.Parameter(torch.randn(1, 1, self.query_dim))
        self.slots_logsigma = torch.nn.Parameter(torch.zeros(1, 1, self.query_dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.init_slots = self.init_slots
        
        '''
        Learn parameter alpha to distinguish signal from background slots
        '''
        self.alpha_mlp = torch.nn.Sequential(
                torch.nn.Linear(query_dim,alpha_depth),
                torch.nn.ReLU(),
                torch.nn.Linear(alpha_depth, 1),
                torch.nn.Sigmoid()
        )
        
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
        
    def build_grid(self):
        '''
        From google slot attention repo:
        https://github.com/nhartman94/google-research/blob/master/slot_attention/model.py#L357C1-L364C53
        '''
        resolution = self.resolution
        xlow, xhigh = self.xlow, self.xhigh
           
        ranges = [np.linspace(xlow, xhigh, num=res) for res in resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="xy")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [resolution[0], resolution[1], -1])
        grid = np.expand_dims(grid, axis=0)
        
        grid = torch.FloatTensor( grid ).to(self.device)
        grid = torch.flatten(grid,1,2)
    
        return grid
                
    def init_slots(self,Nbatch):
        '''
        Slot init taken from
        https://github.com/lucidrains/slot-attention/blob/master/slot_attention/slot_attention.py
        '''
        
        stdhigh, stdlow = self.rlow, self.rhigh
        
        mu = self.slots_mu.expand(Nbatch, self.k_slots, -1)
        sigma = self.slots_logsigma.exp().expand(Nbatch, self.k_slots, -1)
    
        queries = mu + sigma * torch.randn(mu.shape,device=self.device)
    
        # Add the position and scale initialization for the local ref frame
        ref_frame_dim = 3
        pos_scale = torch.rand(Nbatch, self.k_slots, ref_frame_dim,device=self.device)

        pos_scale[:,:2] -= 0.5
        pos_scale[:,-1]  = (stdhigh - stdlow) * pos_scale[:,-1] + stdlow
        
        return queries, pos_scale
     
    def get_keys_vals(self, encoded_data, pos_scale):

        # Get the relative position embedding
        rel_grid = self.abs_grid.unsqueeze(1) - pos_scale[:,:,:2].unsqueeze(2)
        rel_grid /= pos_scale[:,:,-1].unsqueeze(2).unsqueeze(-1)
        
        # Embed it in the same space as the query dimension 
        embed_grid = self.pixel_mult * self.dense( rel_grid )
        
        # keys, vals: (bs, img_dim, query_dim)
        keys = self.toK(encoded_data).unsqueeze(1) + embed_grid
        vals = self.toV(encoded_data).unsqueeze(1) + embed_grid
        
        keys = self.init_mlp(self.queryN(keys))
        vals = self.init_mlp(self.queryN(vals))
        
        return keys, vals
                
    def attention_and_weights(self,queries,keys):
        
        logits = torch.einsum('bse,bsde->bsd',queries,keys) * self.softmax_T
        
        att = torch.nn.functional.softmax(logits, dim = 1)
        
        div = torch.sum(att, dim = -1, keepdims = True)
        wts = att/div + 1e-8
        return att,wts

    def update_frames(self,wts):
        '''
        Update the relative frame position
        '''
        
        # expand to include the batch dim
        grid_exp = self.abs_grid.expand(wts.shape[0],-1,2)
        
        new_pos = torch.einsum('bsd,bde->bse',wts,grid_exp)
        
        new_scale = torch.sum(torch.pow(grid_exp.unsqueeze(1) - new_pos.unsqueeze(2),2),dim=-1)
        
        new_scale = torch.einsum('bsd,bsd->bs', wts, new_scale)
        new_scale = torch.sqrt(new_scale)
        
        return torch.cat([new_pos,new_scale.unsqueeze(-1)],axis=-1)
        
    def iterate(self, queries, pos_scale, encoded_data):
        
        # Get the keys and values in the ref ref frame
        keys, vals = self.get_keys_vals(encoded_data,pos_scale)
        
        # att,wts: (bs, k_slots, img_dim)
        att,wts = self.attention_and_weights(self.queryN(queries),keys)   
        
        new_pos_scale = self.update_frames(wts)
        
        # Update the queries with the recurrent block
        updates = torch.einsum('bsd,bsde->bse',wts,vals) # bs, n_slots, query_dim
        
        updates = self.gru(
            updates.reshape(-1,self.query_dim),
            queries.reshape(-1,self.query_dim),
        )
        
        return updates.reshape(queries.shape), new_pos_scale
        
    def forward(self, data):
    
        '''
        Step 1: Extract the CNN features
        '''
        encoded_data = self.CNN_encoder(data) # Apply the CNN encoder
        encoded_data = torch.permute(encoded_data,(0,2,3,1)) # Put channel dim at the end
        encoded_data = torch.flatten(encoded_data,1,2) # flatten pixel dims
        encoded_data = self.dataN(encoded_data)
        
        '''
        Step 2: Initialize the slots
        '''
        Nbatch = data.shape[0]
        queries, pos_scale = self.init_slots(Nbatch) # Shape (Nbatch, k_slots, query_dim)
                
        '''
        Step 3: Iterate through the reconstruction
        '''
        for i in range(self.n_iter):
            queries, pos_scale = self.iterate(queries, pos_scale, encoded_data)    
            
        # With the final query vector, calc the attn, weights, + rel ref frames
        keys, vals = self.get_keys_vals(encoded_data,pos_scale)
        att, wts = self.attention_and_weights(self.queryN(queries),keys)   
        new_pos_scale = self.update_frames(wts)
        
        # learn alpha (signal vs background slot)
        alpha = self.alpha_mlp(queries)
                
        if self.learn_slot_feat:
            slot_feat = self.final_mlp(queries)
            
            # Want to learn the delta from the previously estimated position
            slot_feat += new_pos_scale
            
            return queries, att, slot_feat, alpha 
        
        else:
            return queries, att, wts, alpha
        

class mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, pre_act):
        super(ResidualBlock, self).__init__()
        if activation == 'mish':
            activation = mish()
        self.activation = activation
        self.pre_act = pre_act
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs, inputs_scaled):
        x = inputs
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.activation(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.pre_act:
            y = self.activation(y)
            
        x = self.conv_res(x)
        x = x + y
        if not self.pre_act:
            x = self.activation(x)
            
        x = torch.cat((x, inputs_scaled), dim=1)
        return x

class Encoder_resnet_S1(nn.Module):
    def __init__(self, nPixels=32, latent_dim=128, nMaxClusters=2, activation=mish(), use_vae=False, pre_act=False, filters=[16,16,32]):
        super(Encoder_resnet_S1, self).__init__()
        #self.initial_conv = nn.Conv2d(3, filters[0], kernel_size=3, stride=1, padding=1)
        self.res_blocks = nn.ModuleList()
        self.activation = activation
        self.use_vae = use_vae
        
        self.res_block11 = ResidualBlock(1,16, activation, pre_act)
        self.res_block12 = ResidualBlock(17,16, activation, pre_act)
        self.res_block13 = ResidualBlock(17,16, activation, pre_act)
        self.res_block14 = ResidualBlock(17,16, activation, pre_act)
        self.res_block15 = ResidualBlock(17,16, activation, pre_act)

        self.res_block21 = ResidualBlock(17,16, activation, pre_act)
        self.res_block22 = ResidualBlock(17,16, activation, pre_act)
        self.res_block23 = ResidualBlock(17,16, activation, pre_act)
        self.res_block24 = ResidualBlock(17,16, activation, pre_act)
        self.res_block25 = ResidualBlock(17,16, activation, pre_act)

        self.res_block31 = ResidualBlock(17,32, activation, pre_act)
        self.res_block32 = ResidualBlock(33,32, activation, pre_act)
        self.res_block33 = ResidualBlock(33,32, activation, pre_act)
        self.res_block34 = ResidualBlock(33,32, activation, pre_act)
        self.res_block35 = ResidualBlock(33,32, activation, pre_act)
        
        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear((filters[-1]+1) * nPixels * nPixels // (2 ** len(filters))**2, 256, bias=False)
       
        self.bn_dense = nn.BatchNorm1d(256)
        #self.sampling_layer = Sampling()

        if use_vae:
            self.z_mean_full = nn.Linear(256, nMaxClusters*latent_dim)
            self.z_log_var_full = nn.Linear(256,nMaxClusters*latent_dim)
        else:
            self.z_full = nn.Linear(256, nMaxClusters*latent_dim)
            
            
        # Sara's adjustment ideas
        self.upsample = nn.ConvTranspose2d(33, 16, 8, stride=8, padding=0) # is this a smart thing to do?
        self.lastdense = nn.Linear(256, 16*32*32)

    def forward(self, inputs):
        x = inputs
        inputs_scaled = inputs
        #inputs_scaled = F.interpolate(x, size=(nPixels, nPixels))

        inputs_scaled = F.interpolate(inputs_scaled, size=(x.shape[2], x.shape[3]))
        x = self.res_block11(x,inputs_scaled)
        x = self.res_block12(x,inputs_scaled)
        x = self.res_block13(x,inputs_scaled)
        x = self.res_block14(x,inputs_scaled)
        x = self.res_block15(x,inputs_scaled)  
        x = self.pooling(x)

        inputs_scaled = F.interpolate(inputs_scaled, size=(x.shape[2], x.shape[3]))
        x = self.res_block21(x,inputs_scaled)
        x = self.res_block22(x,inputs_scaled)
        x = self.res_block23(x,inputs_scaled)
        x = self.res_block24(x,inputs_scaled)
        x = self.res_block25(x,inputs_scaled)
        x = self.pooling(x)

        inputs_scaled = F.interpolate(inputs_scaled, size=(x.shape[2], x.shape[3]))
        x = self.res_block31(x,inputs_scaled)
        x = self.res_block32(x,inputs_scaled)
        x = self.res_block33(x,inputs_scaled)
        x = self.res_block34(x,inputs_scaled)
        x = self.res_block35(x,inputs_scaled)
        x = self.pooling(x)
        
        x = self.upsample(x) # option 1... upsampling with CNN from [bs, 33, 4,4] to [bs, 16, 32, 32]
        return x


class Encoder_resnet_S2(nn.Module):
    def __init__(self, nPixels=32, latent_dim=128, nMaxClusters=2, activation=mish(), use_vae=False, pre_act=False, filters=[16,16,32]):
        super(Encoder_resnet_S2, self).__init__()
        #self.initial_conv = nn.Conv2d(3, filters[0], kernel_size=3, stride=1, padding=1)
        self.res_blocks = nn.ModuleList()
        self.activation = activation
        self.use_vae = use_vae
        
        self.res_block11 = ResidualBlock(1,16, activation, pre_act)
        self.res_block12 = ResidualBlock(17,16, activation, pre_act)
        self.res_block13 = ResidualBlock(17,16, activation, pre_act)
        self.res_block14 = ResidualBlock(17,16, activation, pre_act)
        self.res_block15 = ResidualBlock(17,16, activation, pre_act)

        self.res_block21 = ResidualBlock(17,16, activation, pre_act)
        self.res_block22 = ResidualBlock(17,16, activation, pre_act)
        self.res_block23 = ResidualBlock(17,16, activation, pre_act)
        self.res_block24 = ResidualBlock(17,16, activation, pre_act)
        self.res_block25 = ResidualBlock(17,16, activation, pre_act)

        self.res_block31 = ResidualBlock(17,32, activation, pre_act)
        self.res_block32 = ResidualBlock(33,32, activation, pre_act)
        self.res_block33 = ResidualBlock(33,32, activation, pre_act)
        self.res_block34 = ResidualBlock(33,32, activation, pre_act)
        self.res_block35 = ResidualBlock(33,32, activation, pre_act)
        
        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear((filters[-1]+1) * nPixels * nPixels // (2 ** len(filters))**2, 256, bias=False)
       
        self.bn_dense = nn.BatchNorm1d(256)
        #self.sampling_layer = Sampling()

        if use_vae:
            self.z_mean_full = nn.Linear(256, nMaxClusters*latent_dim)
            self.z_log_var_full = nn.Linear(256,nMaxClusters*latent_dim)
        else:
            self.z_full = nn.Linear(256, nMaxClusters*latent_dim)
            
            
        # Sara's adjustment ideas
        self.upsample = nn.ConvTranspose2d(33, 16, 8, stride=8, padding=0) # is this a smart thing to do?
        self.lastdense = nn.Linear(256, 16*32*32)

    def forward(self, inputs):
        x = inputs
        inputs_scaled = inputs
        #inputs_scaled = F.interpolate(x, size=(nPixels, nPixels))

        inputs_scaled = F.interpolate(inputs_scaled, size=(x.shape[2], x.shape[3]))
        x = self.res_block11(x,inputs_scaled)
        x = self.res_block12(x,inputs_scaled)
        x = self.res_block13(x,inputs_scaled)
        x = self.res_block14(x,inputs_scaled)
        x = self.res_block15(x,inputs_scaled)  
        x = self.pooling(x)

        inputs_scaled = F.interpolate(inputs_scaled, size=(x.shape[2], x.shape[3]))
        x = self.res_block21(x,inputs_scaled)
        x = self.res_block22(x,inputs_scaled)
        x = self.res_block23(x,inputs_scaled)
        x = self.res_block24(x,inputs_scaled)
        x = self.res_block25(x,inputs_scaled)
        x = self.pooling(x)

        inputs_scaled = F.interpolate(inputs_scaled, size=(x.shape[2], x.shape[3]))
        x = self.res_block31(x,inputs_scaled)
        x = self.res_block32(x,inputs_scaled)
        x = self.res_block33(x,inputs_scaled)
        x = self.res_block34(x,inputs_scaled)
        x = self.res_block35(x,inputs_scaled)
        x = self.pooling(x)
        #x = self.upsample(x) # option 1... upsampling with CNN from [bs, 33, 4,4] to [bs, 16, 32, 32]
        #return x
        
        x = self.flatten(x)
        #print(x.shape)
        x = self.dense(x)
        
        x = self.bn_dense(x)
        x = self.activation(x)
        
        # Option 2: just add one last dense layer and reshape it!
        x = self.lastdense(x)
        x = x.reshape([inputs.shape[0], 16, 32, 32]) # horribly hard coded I know
        
        return x


