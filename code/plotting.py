'''
plotting.py

Nicole Hartman
Summer 2023
'''
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Circle

def plot_chosen_slots(losses, mask, att_img, Y_true, Y_pred, color='C0',cmap='Blues',figname=''):
    n_rings = att_img.shape[0]
    fig, axs = plt.subplots(1,n_rings+2,figsize=(3*(n_rings + 2) ,2.5))

    for k,v in losses.items():
        axs[0].plot(v,label=k)
    axs[0].set_xlabel('Iters')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    
    imgs   = [mask] + [att_img[i] for i in range(n_rings)]
    titles = ['Target']+[f'Slot {i}' for i in range(n_rings)]
    extent = [-0.5, 0.5]*2
    for i, (ax,img,title) in enumerate(zip(axs[1:],imgs, titles)):
        
        im = ax.imshow(img.detach().cpu().numpy(),cmap=cmap,
                       extent=extent,origin='lower') #,vmin=0,vmax=1)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        ax.set_title(title)
        

    # Add on the target image
    axi = axs[1]
    c_true = 'r'
    c_pred = 'k'
    for yi in Y_true.cpu().numpy():
    
        axi.scatter(*yi[:2],marker='x',color=c_true)
        circle = Circle(yi[:2],yi[2],fill=False,color=c_true)
        axi.add_patch(circle)
        
        axi.set_xlim(-0.5,0.5)
        axi.set_ylim(-0.5,0.5)
    
    for axi,yi,oi in zip(axs[2:],Y_true.cpu().numpy(),Y_pred.detach().cpu().numpy()):
        
        axi.scatter(*yi[:2],marker='x',color=c_true)
        circle = Circle(yi[:2],yi[2],fill=False,color=c_true)
        axi.add_patch(circle)
        
        axi.scatter(*oi[:2],marker='x',color=c_pred)
        circle = Circle(oi[:2],oi[2],fill=False,color=c_pred)
        axi.add_patch(circle)

        axi.set_xlim(-0.5,0.5)
        axi.set_ylim(-0.5,0.5)
        
    if figname:
        plt.savefig(figname)

    plt.show()
    plt.close()

def plot_kslots(losses, mask, att, k_slots, color='C2',cmap='Greens',figname=''):
    
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
    plt.show()
    plt.close()

def plot_kslots_iters(model, data, iEvt, color='C2',cmap='Greens',figname=''):
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
    plt.show()
    plt.close()


def plot_kslots_grads(model,grads, iEvt, color='C2',cmap='Greens',figname=''):
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
    plt.show()
    plt.close()

def plot_slots_with_alpha(losses, X, att_signal, Y, cmap = 'Blues', figname=''):
    X = X.detach().cpu()
    Y = Y.detach().cpu()
    att_signal = att_signal.detach().cpu()
    N_obj = Y.shape[0]
    fig, axs = plt.subplots(1,3+N_obj, figsize=(3*(N_obj + 3) ,3))

    # loss
    for k,v in losses.items():
        if (len(v)!=0):
            axs[0].plot(v,label=k)
    axs[0].set_xlabel('Iters')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].set_box_aspect(1)

    # truth
    extent= [-0.5, 0.5]*2

    img = axs[1].imshow(X[0].detach().cpu(), origin='lower', cmap=cmap, extent=extent)
    axs[1].set_title("Truth")
    # colorbar:
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img, cax=cax, orientation='vertical')

    # rings
    for j in range(1, N_obj+3):
        axs[j].scatter(Y[:, 0], Y[:, 1], marker="x", c='r')
        for yi in Y.cpu().numpy():
            circle = Circle(yi[:2],yi[2],fill=False,color='r')
            axs[j].add_patch(circle)
        axs[j].set_ylim(extent[0], extent[1])
        axs[j].set_xlim(extent[0], extent[1])

    # reco
    for k, j in enumerate(range(2, 3+N_obj)):
        axs[j].imshow(att_signal.numpy()[k],  origin='lower', cmap=cmap, extent=extent)
        axs[j].set_title(r"$\alpha \cdot \mathrm{att}$"+" (slot {})".format(k))
        # colorbar:
        divider = make_axes_locatable(axs[j])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(img, cax=cax, orientation='vertical')

    plt.tight_layout()
    if figname:
        plt.savefig(figname)
    plt.show()
    plt.close()