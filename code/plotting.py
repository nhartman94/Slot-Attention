'''
plotting.py

Nicole Hartman
Summer 2023
'''
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_kslots(losses, mask, att, k_slots, color='C2',cmap='Greens',figname='',showImg=True):
    
    fig, axs = plt.subplots(1,k_slots+2,figsize=(2.75 * (k_slots + 2) ,2.5))

    axs[0].plot(losses,color=color)
    axs[0].set_xlabel('Iters')
    axs[0].set_ylabel('Loss')
    
    imgs   = [mask] + [att[i] for i in range(k_slots)]
    titles = ['Target']+[f'Slot {i}' for i in range(k_slots)]
    
    for i, (ax,img,title) in enumerate(zip(axs[1:],imgs, titles)):
        
        im = ax.imshow(img,cmap=cmap,vmin=0,vmax=1)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        ax.set_title(title)

        ax.axis('off')

    if figname:
        plt.savefig(figname)
    if showImg:
        plt.show()
    plt.close()

def plot_kslots_iters(model, data, iEvt, color='C2',cmap='Greens',figname='',showImg=True):
    '''
    Plot the attention masks across the iterations
    '''
    
    n_iter = model.n_iter
    k_slots = model.k_slots
    nPixels, nPixels = model.resolution

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

    if figname:
        plt.savefig(figname)  
    if showImg:
        plt.show()
    plt.close()


def plot_kslots_grads(model,grads, iEvt, color='C2',cmap='Greens',figname='',showImg=True):
    '''
    Plot the gradients across the attention maps
    '''
    
    n_iter = model.n_iter
    k_slots = model.k_slots
    nPixels,nPixels=model.resolution

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

    if figname:
        plt.savefig(figname)    
    if showImg:
        plt.show()
    plt.close()