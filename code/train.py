
'''
Script for training the slot attention models.
Starting from models and functions in Lukas's notebook.

Nicole Hartman
Summer 2023
'''

# ML packages
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# custom code for this repo
from model import InvariantSlotAttention, SlotAttentionPosEmbed
from data import make_batch
from plotting import plot_chosen_slots, plot_kslots, plot_kslots_iters, plot_kslots_grads

# file IO packages
import os
import yaml, json
from glob import glob
from argparse import ArgumentParser

def hungarian_matching(pairwise_cost):
    '''
    Input:
    - pairwise_cost


    Hungarian section Translated from the TensorFlow loss function (from 2006.15055 code):
    https://github.com/nhartman94/google-research/blob/master/slot_attention/utils.py#L26-L57
    '''
    
    indices = list(map(linear_sum_assignment, pairwise_cost.cpu()))
    indices = torch.LongTensor(np.array(indices))
    
    loss = 0
    for pi,(ri,ci) in zip(pairwise_cost,indices):
        loss += pi[ri,ci].sum()
    
    return indices 

def expDecaySchedule(base_learning_rate, Ntrain,warmup_steps,decay_rate,decay_steps):
    '''
    Exponential weight deceay (w/ linear warmup)
    '''

    xx = np.linspace(0,Ntrain,Ntrain)

    lr = base_learning_rate * np.power(decay_rate, xx / decay_steps)
    lr *= np.where(xx < warmup_steps, xx/warmup_steps,1)

    return lr

def cosineSchedule(base_learning_rate, T, warmup_steps):
    '''
    Cosine dcay (w/ linear warm up)

    Inputs:
    - base_learning_rate: max lr
    - T: Maximum number of training steps
    - warm_up steps
    
    Output:
    - lr: vector of len T of the lr at each iter
    
    '''
    
    xx = np.linspace(0,T,T)

    lr = base_learning_rate
    lr *= .5 * (1 + np.cos(xx * np.pi / T))
    lr *= np.where(xx < warmup_steps, xx/warmup_steps,1)

    return lr


def comb_loss(att,flat_mask,Y=None,Y_pred=None,alpha=1):
    '''
    Goal: Given a NN that predicts both an occupancy mask
    and a center and radius for each slot, combine these terms 
    into a combined loss function:
    
    L = L_bce + alpha * L_mse
    
    Note: This function should be general enough to either calculate
    the losses of all the combinations of slots and targets or just
    the single loss between the loss and the chosen target
        
    '''
    
    max_n_rings = flat_mask.shape[1]
    k_slots = att.shape[1]
    
    att_ext  = torch.tile(att.unsqueeze(2), dims=(1,1,max_n_rings,1)) 
    mask_ext = torch.tile(flat_mask.unsqueeze(1),dims=(1,k_slots,1,1)) 
    
    l_bce = F.binary_cross_entropy(att_ext,mask_ext,reduction='none').mean(axis=-1)
    
    if alpha == 0:
        return l_bce
    
    else:
    
        # Calc MSEmse_loss(Y,Y_pred)
        l_mse = torch.nn.MSELoss(reduction='none')(Y_pred.unsqueeze(2), Y.unsqueeze(1)).mean(axis=-1)
        return l_bce + alpha * l_mse
    

def train(model, 
          Ntrain = 5000, 
          bs=32, 
          lr=3e-4,
          warmup_steps=5_000,
          alpha=1,
          losses = {'tot':[],'bce':[],'mse':[]},
          kwargs={'isRing': True, 'N_clusters':2},
          clip_val = 1,
          device='cpu',
          plot_every=250, 
          save_every=1000,
          color='C0',cmap='Blues',
          modelDir='.',figDir=''):
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
    - alpha (default 1): The weight to add to the MSE slot (x,y,r) predictor function
        loss = l_bce + alpha * mse
    - losses: Dict for the `tot`, `bce`, and `mse` loss (terms)
        If starting from a warm-start, a list of the previous losses to append to
    - kwargs: dictionary of key word arguments to pass to the `make_batch` function
    - clip_val: Value to use for "gradient clipping" (default 1)
    - device: (default cpu) for data loading, needs to be the same as model
    - plot_every (default 250), save_every (default 1000): How often to plot / save the model
    - color,cmap: options that get passed the diagonistic plotting scripts
    - modelDir: Directory to save the models to as
        {modelDir}/m_{iter}.pt
    - figDir: Directory to save the figures to 
    '''

    # Learning rate schedule config
    base_learning_rate = lr
    
    opt = torch.optim.Adam(model.parameters(), base_learning_rate)
    model.train()
    
    k_slots = model.k_slots
    max_n_rings = kwargs['N_clusters']
    resolution = model.resolution
    kwargs['device'] = device

    start = len(losses)
    for i in range(start,start+Ntrain):

        learning_rate = base_learning_rate * 0.5 * (1 + np.cos(np.pi * i / Ntrain))
        if i < warmup_steps:
            learning_rate *= (i / warmup_steps)
        
        opt.param_groups[0]['lr'] = learning_rate
        
        X, Y, mask = make_batch(N_events=bs, **kwargs)
        
        queries, att, Y_pred = model(X)
         
        # Reshape the target mask to be flat in the pixels (same shape as att)
        flat_mask = mask.reshape(-1,max_n_rings, np.prod(resolution))      
        with torch.no_grad():
            
            att_ext  = torch.tile(att.unsqueeze(2), dims=(1,1,max_n_rings,1)) 
            mask_ext = torch.tile(flat_mask.unsqueeze(1),dims=(1,k_slots,1,1)) 
            
            pairwise_cost = F.binary_cross_entropy(att_ext,mask_ext,reduction='none').mean(axis=-1)
            
            # pairwise_cost = comb_loss(att,flat_mask,Y,Y_pred,alpha)
            indices = hungarian_matching(pairwise_cost)

        # Apply the sorting to the predict
        bis=torch.arange(bs).to(device)
        indices=indices.to(device)

        # Loss calc
        slots_sorted = torch.cat([att[bis,indices[:,0,ri]].unsqueeze(1) for ri in range(max_n_rings)],dim=1)
        rings_sorted = torch.cat([flat_mask[bis,indices[:,1,ri]].unsqueeze(1) for ri in range(max_n_rings)],dim=1)
        l_bce = F.binary_cross_entropy(slots_sorted,rings_sorted,reduction='none').sum(axis=1).mean()
        
        Y_pred_sorted = torch.cat([Y_pred[bis,indices[:,0,ri]].unsqueeze(1) for ri in range(max_n_rings)],dim=1)
        Y_true_sorted = torch.cat([Y[bis,indices[:,1,ri]].unsqueeze(1) for ri in range(max_n_rings)],dim=1)

        l_mse = torch.nn.MSELoss(reduction='none')(Y_pred_sorted,Y_true_sorted).sum(axis=1).mean()
    
        # Calculate the loss
        li = l_bce + alpha*l_mse
        
        li.backward()
        clip_val=1
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        
        opt.step()
        opt.zero_grad()

        losses['tot'].append(float(li))
        losses['bce'].append(float(l_bce))
        losses['mse'].append(float(l_mse))
        
        if i % plot_every == 0:
            print('iter',i,', loss',li.detach().cpu().numpy(),', lr',opt.param_groups[0]['lr'])  
            iEvt = 0

            # losses, mask, att_img, Y_true, Y_pred
            plot_chosen_slots(losses,
                              mask[iEvt].sum(axis=0), 
                              slots_sorted[iEvt].reshape(max_n_rings,*resolution),
                              Y_true_sorted[iEvt],
                              Y_pred_sorted[iEvt],
                              figname=f'{figDir}/loss_chosen_slots.jpg')
            
        if (i % save_every == 0) and modelDir:
            torch.save(model.state_dict(), f'{modelDir}/m_{i}.pt')
            with open(f'{modelDir}/loss.json','w') as f:
                json.dump(losses, f)

    model.eval()
    return model,losses

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/needToDefine.json",
        help="hp configs for the model and optimizer",
    )

    parser.add_argument(
        "--device",
        dest="device",
        type=str,
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

    # Parameters to start from a previous training
    parser.add_argument(
        "--warm_start",
        action='store_true',
        help="Whether to start this training from some previous weights"
    )

    parser.add_argument(
        "--iter_to_load",
        type=int,
        default=-1,
        help="Which iteration to load in for the warm start training.\n"\
            +"Default (-1) will load in the last available model.\n"\
            +"Only used if `--warm_start` is passed."
    )

    parser.add_argument(
        "--warm_start_config",
        type=str,
        default="",
        help="The config file to start from for training with the warm start flag.\n"\
            +"If this argument is not passed, will load in the model weights from\n" \
            +"the last iteration of the file passed with the --config flag"
    )


    args = parser.parse_args()

    config = args.config
    device = args.device
    plot_every = args.plot_every
    save_every = args.save_every

    warm_start = args.warm_start
    iter_to_load = args.iter_to_load
    warm_start_config = args.warm_start_config

    # Open up the configs file to retreive the parameters
    with open(config, "r")as cfile:
        cdict = yaml.safe_load(cfile)
    
    opt = cdict["opt"]
    hps = cdict["hps"]
    kwargs = cdict["data"] # to pass to the data loading fct

    # Check that we haven't trained these config files before
    modelID = config.replace('configs','').replace('.yaml','') 
    modelDir = f'models/{modelID}'
    figDir = f'figures/{modelID}'
    
    for newDir in [modelDir, figDir]: 
        try:
            os.mkdir(newDir)
        except FileExistsError:
            print(newDir,'already exists')

    # Define the architecture 
    hps['device'] = device # the model also needs to 

    model = InvariantSlotAttention(**hps).to(device)

    # Load in the weights 
    if warm_start:

        prev_config = warm_start_config if warm_start_config else config
        prevID = prev_config.replace('configs','').replace('.yaml','') 

        print(f'Starting from an earlier training from {prev_config}')

        if iter_to_load == -1:
            # Check for what was the last training iteration
            modelChkpts = glob(f'models/{prevID}/m_*.pt')
            if len(modelChkpts) == 0:
                print('ERROR -- No files fround for',modelChkpts, 'when requesting to train from warm_start')
                raise FileNotFoundError

            savedIters = [mName.split('/')[-1].split('.')[0].split('_')[-1] for mName in modelChkpts   ]
            savedIters = np.array([int(i) for i in savedIters])                     
            lastIter = np.max(savedIters)

            modelToLoad = f'models/{prevID}/m_{lastIter}.pt'
        else:
            modelToLoad = f'models/{prevID}/m_{iter_to_load}.pt'
            if not os.path.exists(modelToLoad):
                print('Error asking to load',modelToLoad,'but it doesn\'t exist. Returning.')

        # Load in the weights
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        # for the "save on gpu, load on gpu" cmd
        model.load_state_dict(torch.load(modelToLoad,map_location=device),strict=False)
        model.to(device)
        
        # Also load in the losses if we're starting from the same config file
        if len(warm_start_config) == 0:
            with open(f'models/{prevID}/loss.json') as f:
                losses = json.load(f)
        else: 
            losses = {'tot':[],'bce':[],'mse':[]}

    else: 
        print('Training starting from freshly initialized weights')
        losses = {'tot':[],'bce':[],'mse':[]}

    # Train the model 
    train(model, 
          **opt,
          losses=losses,
          N_obj=kwargs,
          device=device,
          plot_every=plot_every, 
          save_every=save_every,
          modelDir=modelDir,figDir=figDir)
