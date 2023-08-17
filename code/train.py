'''
Script for training the slot attention models.
Starting from models and functions in Lukas's notebook.

Nicole Hartman
Summer 2023
'''

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from model import SlotAttentionPosEmbed
from data import make_batch
from plotting import plot_kslots, plot_kslots_iters, plot_kslots_grads
import os
import yaml

def hungarian_matching(att, mask,bs, k_slots,max_n_rings,nPixels):
    '''
    Hungarian section Translated from the TensorFlow loss function (from 2006.15055 code):
    https://github.com/nhartman94/google-research/blob/master/slot_attention/utils.py#L26-L57
    '''
    
    flat_mask = mask.reshape(-1,max_n_rings, nPixels*nPixels)[:,None,:,:]
    
    att_ext  = torch.tile(att.unsqueeze(2),  dims=(1,1,max_n_rings,1)) #.reshape(bs * k_slots * max_n_rings , nPixels**2)
    mask_ext = torch.tile(flat_mask,dims=(1,k_slots,1,1)) #.reshape(bs * k_slots * max_n_rings , nPixels**2)
    
    pairwise_cost = F.binary_cross_entropy(att_ext,mask_ext,reduction='none').mean(axis=-1)
    #pairwise_cost = pairwise_cost.reshape(bs, k_slots, max_n_rings)
    
    indices = list(map(linear_sum_assignment, pairwise_cost.cpu()))
    indices = torch.LongTensor(indices)
    
    loss = 0
    for pi,(ri,ci) in zip(pairwise_cost,indices):
        loss += pi[ri,ci].sum()
    
    return indices 


def train(model, 
          Ntrain = 5000, 
          bs=32, 
          lr=3e-4,
          warmup_steps=10_000,
          decay_rate = 0.5,
          decay_steps = 100_000,
          kwargs={},
          device='cpu',
          plot_every=250, 
          save_every=1000,
          color='C0',cmap='Blues',
          modelDir='',figDir='',pl):
    '''
    train
    (Function from Lukas)
    -----------
    
    - model: A pytorch NN model
    - Ntrain: # of training iterations 
    - bs: batch size to use for the sampling
    - lr: The base learning rate for training with adam
        (although this learning rate gets modified w/ the subsequent optimizer schedules)
    - warmup_steps: steps over which you gradually ramp up the learning rate
        If 0 is passed, the warm-up will never be activated
    - decay_rate: factor for an exponential decay schedule:
           lr * decay_rate^(step / decay_steps)
         If the decay rate is set to 1, no dec
    - decay_steps: Controls the timescale of the learning rate decay (see above)
    - device: (default cpu) for data loading... needs to be the same as model
    - color,cmap: options that get passed the diagonistic plotting scripts
    - modelDir: Directory to save the models to as
        {modelDir}/m_{iter}.pt
    - figDir: Directory to save the figures to
    '''

    # Learning rate schedule config
    base_learning_rate = lr
    
    opt = torch.optim.Adam(model.parameters(), base_learning_rate)
    model.train()
    losses = []
    
    for i in range(Ntrain):

        learning_rate = base_learning_rate * decay_rate ** (i / decay_steps)
        if i < warmup_steps:
            learning_rate *= (i / warmup_steps)
        
        opt.param_groups[0]['lr'] = learning_rate
        
        X, Y, mask = make_batch(N_events=bs, **kwargs)
        
        queries, att, wts = model(X)
            
        with torch.no_grad():
            indices = hungarian_matching(att,mask,bs,model.k_slots,max_n_rings,nPixels)

        # Apply the sorting to the predict
        bis=torch.arange(bs).to(device)
        indices=indices.to(device)

        slots_sorted = torch.cat([att[bis,indices[:,0,ri]].unsqueeze(1) for ri in range(max_n_rings)],dim=1)
        
        flat_mask = mask.reshape(-1,max_n_rings, prod(model.resolution))
        rings_sorted = torch.cat([flat_mask[bis,indices[:,1,ri]].unsqueeze(1) for ri in range(max_n_rings)],dim=1)

        # Calculate the loss
        loss = F.binary_cross_entropy(slots_sorted,rings_sorted,reduction='none').sum(axis=1).mean()
        
        loss.backward()
        opt.step()
        opt.zero_grad()

        losses.append(float(loss))
        
        if i % 250 == 0:
            print('iter',i,', loss',loss.detach().cpu().numpy(),', lr',opt.param_groups[0]['lr'])
            
            iEvt = 0
            att_img = att[iEvt].reshape(model.k_slots,nPixels,nPixels)
            plot_kslots(losses, 
                        mask[iEvt].sum(axis=0).detach().cpu().numpy(), 
                        att_img.detach().cpu().numpy(),
                        k_slots, color=color,cmap=cmap)
            
            
            plot_kslots_iters(model, X, iEvt=0)
            plot_kslots_grads(model.gradients,iEvt=0)

    model.eval()
    return model,losses

if __name__ == "main":

    parser = makeParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/needToDefine.json",
        help="hp configs for the model and optimizer",
    )

    parser.add_argument(
        "--device",
        dest="device",
        type="str",
        default='cuda:0'
    )

    parser.add_argument(
        "--plot_every",
        type=int,
        default=250,
        help="How many iteration steps to make between making the diagnostic plots"
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=1000,
        help="How many iteration steps to make between saving model checkpoints"
    )

    args = parser.parse_args()

    config = args.config
    device = args.device
    plot_every = args.plot_every
    save_every = args.save_every

    # Step 1: Retrieve the data loading fct(?)


    # Open up the configs file to retreive the parameters
    with open(config, "r") as cfile:
        cdict = json.load(cfile)

    opt = cdict["opt"]
    hps = cdict["hps"]

    # Check that we haven't trained these config files before
    modelID = config.replace('configs','').replace('.yaml','') 
    if os.path.exists(f'models/{modelID}'):
        print('You ar')


    # Define the architecture 
    model = SlotAttention(k_slots=k_slots,device=device,hidden_dim=16,query_dim=16).to(device)


    # Train the model 
    # ?



