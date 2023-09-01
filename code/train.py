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
from model import SlotAttentionPosEmbed
from data import make_batch
from plotting import plot_kslots, plot_kslots_iters, plot_kslots_grads

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


def train(model, 
          Ntrain = 5000, 
          bs=32, 
          lr=3e-4,
          warmup_steps=5_000,
          decay_rate = 0.5,
          decay_steps = 50_000,
          losses = [],
          kwargs={'isRing': True, 'N_clusters':2},
          device='cpu',
          plot_every=250, 
          save_every=1000,
          color='C0',cmap='Blues',
          modelDir='.',figDir='',showImg=True):
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
    - losses: If starting from a warm-start, a list of the previous losses to append to
    - kwargs: dictionary of key word arguments to pass to the `make_batch` function
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
    
    
    k_slots = model.k_slots
    resolution = model.resolution
    kwargs['device'] = device

    max_n_rings = kwargs['N_clusters']
    isRing = kwargs["isRing"]
    print(f'Training model with {k_slots} slots on {max_n_rings}'+ ("rings" if isRing else "blobs"))

    start = len(losses)
    for i in range(start,start+Ntrain):

        learning_rate = base_learning_rate * decay_rate ** (i / decay_steps)
        if i < warmup_steps:
            learning_rate *= (i / warmup_steps)
        
        opt.param_groups[0]['lr'] = learning_rate
        
        X, Y, mask = make_batch(N_events=bs, **kwargs)
        
        queries, att, wts = model(X)

        with torch.no_grad():

            # Calculate the loss of _all_ possible combinations  
            flat_mask = mask.reshape(-1,max_n_rings, np.prod(resolution))[:,None,:,:]
        
            att_ext  = torch.tile(att.unsqueeze(2),  dims=(1,1,max_n_rings,1)) 
            mask_ext = torch.tile(flat_mask,dims=(1,k_slots,1,1)) 

            pairwise_cost = F.binary_cross_entropy(att_ext,mask_ext,reduction='none').mean(axis=-1)

            indices = hungarian_matching(pairwise_cost)

        # Apply the sorting to the predict
        bis=torch.arange(bs).to(device)
        indices=indices.to(device)

        slots_sorted = torch.cat([att[bis,indices[:,0,ri]].unsqueeze(1) for ri in range(max_n_rings)],dim=1)
        
        flat_mask = mask.reshape(-1,max_n_rings, np.prod(resolution))
        rings_sorted = torch.cat([flat_mask[bis,indices[:,1,ri]].unsqueeze(1) for ri in range(max_n_rings)],dim=1)

        # Calculate the loss
        loss = F.binary_cross_entropy(slots_sorted,rings_sorted,reduction='none').sum(axis=1).mean()
        
        loss.backward()
        opt.step()
        opt.zero_grad()

        losses.append(float(loss))
        
        if i % plot_every == 0:
            print('iter',i,', loss',loss.detach().cpu().numpy(),', lr',opt.param_groups[0]['lr'])
            
            iEvt = 0
            att_img = att[iEvt].reshape(model.k_slots,*resolution)

            plot_kslots(losses, 
                        mask[iEvt].sum(axis=0).detach().cpu().numpy(), 
                        att_img.detach().cpu().numpy(),
                        k_slots, color=color,cmap=cmap,
                        figname=f'{figDir}/loss-slots-iter{i}-evt{iEvt}.jpg' if figDir else '',
                        showImg=showImg)
            
            
            # plot_kslots_iters(model, X, iEvt=0, color=color,cmap=cmap, 
            #                   figname=f'{figDir}/slots-unroll-iter{i}-evt{iEvt}.jpg',showImg=showImg)
            # plot_kslots_grads(model,model.gradients,iEvt=0, color=color,cmap=cmap,
            #                   figname=f'{figDir}/grad-unroll-iter{i}-evt{iEvt}.jpg',showImg=showImg)

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

    if not os.path.exists(f'models/{modelID}'):
        for newDir in [modelDir, figDir]: 
            os.mkdir(newDir)

    # Define the architecture 
    hps['device'] = device # the model also needs to 
    model = SlotAttentionPosEmbed(**hps).to(device)

    # Load in the weights 
    if warm_start:

        prev_config = warm_start_config if warm_start_config else config
        prevID = prev_config.replace('configs','').replace('.yaml','') 

        print(f'Starting from an earlier training from {prev_config}')

        # Check for what was the last training iteration
        modelChkpts = glob(f'models/{prevID}/m_*.pt')
        if len(modelChkpts) == 0:
            print('ERROR -- No files fround for',modelChkpts, 'when requesting to train from warm_start')
            raise FileNotFoundError

        savedIters = [mName.split('/')[-1].split('.')[0].split('_')[-1] for mName in modelChkpts   ]
        savedIters = np.array([int(i) for i in savedIters])                     
        lastIter = np.max(savedIters)

        modelToLoad = f'models/{prevID}/m_{lastIter}.pt'

        # Load in the weights
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        # for the "save on gpu, load on gpu" cmd
        model.load_state_dict(torch.load(modelToLoad))
        model.to(device)
        
        # Also load in the losses if we're starting from the same config file
        if len(warm_start_config) == 0:
            with open(f'models/{prevID}/loss.json') as f:
                losses = json.load(f)
        else: 
            losses = []

    else: 
        print('Training starting from freshly initialized weights')
        losses = []

    # Train the model 
    train(model, 
          **opt,
          losses=losses,
          kwargs=kwargs,
          device=device,
          plot_every=plot_every, 
          save_every=save_every,
          modelDir=modelDir,figDir=figDir)
