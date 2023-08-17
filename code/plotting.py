'''
plotting.py

Nicole Hartman
Summer 2023
'''
import torch

def train(model, Ntrain = 5000, bs=32, device='cpu', color='C2',cmap='Greens'):
    '''
    train
    -----------
    
    - model
    - Ntrain: # of training iterations
    - color,cmap -- options that get passed the
    '''

    # Learning rate schedule config
    base_learning_rate = 3e-4
    
    decay_rate = 0.5
    warmup_steps=10_000
    decay_steps = 100_000
    
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
        
        flat_mask = mask.reshape(-1,max_n_rings, nPixels*nPixels)
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

def plot_kslots_iters(model, data, iEvt, color='C2',cmap='Greens'):
    '''
    Plot the attention masks across the iterations
    '''
    
    n_iter = model.n_iter
    k_slots = model.k_slots
    
    attn_masks = []

    with torch.no_grad():
        # Run through the model code to eval the attn masks
        queries = model.init_slots(data.shape[0]) 
        encoded_data = model.encoder(data)

        for i in range(n_iter):
            # Get the mask
            att,wts = model.attention_and_weights(model.queryN(queries), encoded_data) 
            attn_masks.append(att.detach().cpu())

            # Get the updated query
            queries = model.iterate(queries, encoded_data)

        # Get the final mask
        att,wts = model.attention_and_weights(model.queryN(queries), encoded_data) 
        attn_masks.append(att.detach().cpu())
    
    '''
    Make the plot
    '''
    fig, axs = plt.subplots(n_iter+1,k_slots,figsize=(2.75*k_slots,2.5*(n_iter+1)))
 
    for i, (ax_i, att) in enumerate(zip(axs, attn_masks)):
        
        att_img = att[iEvt].reshape(k_slots,nPixels,nPixels)
        
        for j, (ax, img) in enumerate(zip(ax_i,att_img)):
        
            im = ax.imshow(img,cmap=cmap)#,vmin=0,vmax=1)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

            ax.set_title(f'T={i}: Slot {j}')

            ax.axis('off')
        
    plt.show()


def plot_kslots_grads(grads, iEvt, color='C2',cmap='Greens'):
    '''
    Plot the gradients across the attention maps
    '''
    
    n_iter = model.n_iter
    k_slots = model.k_slots
    
    '''
    Make the plot
    '''
    fig, axs = plt.subplots(n_iter+1,k_slots,figsize=(2.75*k_slots,2.5*(n_iter+1)))
 
    for i, ax_i, att in zip(range(n_iter+1)[::-1],axs, grads):
        
        att_img = att[iEvt].reshape(k_slots,nPixels,nPixels)
        
        for j, (ax, img) in enumerate(zip(ax_i,att_img)):
        
            im = ax.imshow(img.cpu().numpy(),cmap=cmap)#,vmin=0,vmax=1)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

            ax.set_title(f'T={i}: Grad attn {j}')

            ax.axis('off')
        
    plt.show()